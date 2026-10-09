#!/usr/bin/env python3
"""Train the icefall Zipformer-S streaming model on this repo's metadata.

Stage 1, Dataset.ET (42 h), from scratch:

    python train_zipformer.py --config configs/zipformer_s_streaming_et.yaml

Stage 2, after that run has a dev number you want to keep. Weights only,
new ScaledAdam and a new Eden schedule:

    python train_zipformer.py --config configs/zipformer_s_streaming_pool.yaml

The encoder, stateless decoder, joiner, pruned RNN-T, and CTC head are the
icefall modules in third_party/icefall. See third_party/icefall/SOURCE.
"""

from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.dataset import DurationBatchSampler, collate
from decoding.greedy import greedy_decode
from decoding.zipformer_greedy import greedy_search_batch
from evaluation.wer import score_texts
from models.zipformer_icefall import (
    build_zipformer,
    combine_loss,
    eden,
    encoder_frames,
    labels_ok,
    load_icefall,
    loss_scales,
    ragged_labels,
    scaled_adam,
)
from train import (
    _amp_setup,
    _capture_rng,
    _load_history_file,
    _merge_history,
    _restore_rng,
    _save_history,
    copy_init_sidecars,
    load_cmvn,
    load_tokenizer,
    make_loader,
)
from utils import ROOT, load_yaml, save_json, set_seed, setup_logging

log = setup_logging()


def set_batch_count(model: torch.nn.Module, batch_count: float) -> None:
    """icefall ``zipformer/train.py`` ``set_batch_count``.

    ScheduledFloat dropout and the whitener read ``module.batch_count``.
    Names are set on the first call; the module list is cached because walking
    ``named_modules`` every step cost ~13 ms of Python.
    """
    targets = model.__dict__.get("_batch_count_targets")
    if targets is None:
        targets = []
        for name, module in model.named_modules():
            if hasattr(module, "name"):
                module.name = name
            if hasattr(module, "batch_count"):
                targets.append(module)
        model.__dict__["_batch_count_targets"] = targets
    for module in targets:
        module.batch_count = batch_count


def adjusted_batch_count(batch_idx: int, batch_seconds: float, ref_duration: float) -> float:
    """icefall ``get_adjusted_batch_count`` at world size 1.

    Dropout schedules advance by audio seconds relative to ``ref_duration``
    (600 s in the LibriSpeech recipe), not by optimizer steps.
    """
    return batch_idx * batch_seconds / ref_duration


def spec_augment(feats: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Mask sizes from ``get_spec_augment`` in ``zipformer/train.py``.

    ``time_mask_ratio`` defaults to 1, so 10 time masks, each at most 100
    frames and at most 15% of the utterance, plus 2 frequency masks of width
    27. Time warp stays off (``time_warp_factor=0`` in that function).

    Masks are drawn on the device for the whole batch. The spans match the
    old per-utterance loop; the random draws do not.
    """
    out = feats.clone()
    b, t, f = out.shape
    ti = lengths.to(device=out.device, dtype=torch.long).clamp(min=1, max=t)
    max_frames = torch.minimum(
        torch.full_like(ti, 100),
        (ti.to(torch.float32) * 0.15).to(torch.long).clamp(min=1),
    )
    width = (torch.rand(10, b, device=out.device) * (max_frames.to(torch.float32) + 1)).to(torch.long)
    span = (ti - width + 1).clamp(min=1)
    t0 = (torch.rand(10, b, device=out.device) * span.to(torch.float32)).to(torch.long)
    valid = (width > 0) & (ti > width)
    frames = torch.arange(t, device=out.device).view(1, 1, t)
    time_hit = (
        (frames >= t0[:, :, None])
        & (frames < (t0 + width)[:, :, None])
        & valid[:, :, None]
    ).any(dim=0)
    out.masked_fill_(time_hit[:, :, None], 0)

    freq_width = torch.randint(0, 28, (2, b), device=out.device)
    freq_span = (f - freq_width + 1).clamp(min=1)
    f0 = (torch.rand(2, b, device=out.device) * freq_span.to(torch.float32)).to(torch.long)
    freq_ok = (freq_width > 0) & (f > freq_width)
    bins = torch.arange(f, device=out.device).view(1, 1, f)
    freq_hit = (
        (bins >= f0[:, :, None])
        & (bins < (f0 + freq_width)[:, :, None])
        & freq_ok[:, :, None]
    ).any(dim=0)
    out.masked_fill_(freq_hit[:, None, :], 0)
    return out


def _drop_short_labels(dataset, tokenizer, split: str) -> int:
    """Drop clips that fail the CTC check in ``remove_short_and_long_utt``."""
    keep = []
    dropped = 0
    for row in dataset.rows:
        n = len(tokenizer.encode(row["text"]))
        frames = int(round(float(row.get("duration_s") or 0.0) * 100))
        if frames >= 7 and labels_ok(frames, n):
            keep.append(row)
        else:
            dropped += 1
    dataset.rows = keep
    if dropped:
        log.info("%s: dropped %d clips with fewer encoder frames than tokens", split, dropped)
    return dropped


def _rebuild(loader: DataLoader, cfg: dict, train: bool) -> DataLoader:
    tcfg = cfg["train"]
    n_workers = int(tcfg.get("num_workers", 4))
    kw: dict = {"collate_fn": collate, "num_workers": n_workers, "pin_memory": True}
    if n_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = int(tcfg.get("prefetch_factor", 4))
    ds = loader.dataset
    if train:
        sampler = DurationBatchSampler(
            ds, max_seconds=float(tcfg.get("batch_seconds", 80)), shuffle=True, seed=int(cfg.get("seed", 42))
        )
        return DataLoader(ds, batch_sampler=sampler, **kw)
    return DataLoader(ds, batch_size=int(tcfg.get("eval_batch_size", 8)), shuffle=False, **kw)


def _start_waveform_augment(loader: DataLoader, augment, epoch: int, start_epoch: int, cfg: dict) -> DataLoader:
    """Turn ``train.augment`` on once ``epoch`` reaches ``train.augment_start_epoch``.

    Persistent workers keep their own copy of the dataset, so setting the
    attribute is not enough; the loader is rebuilt and keeps its shuffle seed.
    """
    ds = loader.dataset
    if augment is None or ds.augment is not None or epoch < start_epoch:
        return loader
    ds.augment = augment
    seed = loader.batch_sampler.seed
    loader = _rebuild(loader, cfg, True)
    loader.batch_sampler.seed = seed
    log.info("epoch %d: waveform augmentation on", epoch)
    return loader


def _keep_indices(feat_lens: torch.Tensor, target_lens: torch.Tensor) -> torch.Tensor:
    ok = []
    for i in range(feat_lens.numel()):
        if labels_ok(int(feat_lens[i]), int(target_lens[i])):
            ok.append(i)
    if not ok:
        return feat_lens.new_zeros((0,), dtype=torch.long)
    return torch.tensor(ok, device=feat_lens.device, dtype=torch.long)


def _take(batch: dict, index: torch.Tensor) -> dict:
    cpu_index = index.detach().cpu()
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.index_select(0, index if value.device == index.device else cpu_index)
        elif isinstance(value, list):
            ids = cpu_index.tolist()
            out[key] = [value[i] for i in ids]
        else:
            out[key] = value
    return out


def _forward_loss(model, feats, lens, targets, target_lens, tcfg, batch_idx: int):
    y = ragged_labels(targets, target_lens)
    simple, pruned, ctc, _, _ = model(
        x=feats,
        x_lens=lens,
        y=y,
        prune_range=int(tcfg.get("prune_range", 5)),
        am_scale=float(tcfg.get("am_scale", 0.0)),
        lm_scale=float(tcfg.get("lm_scale", 0.25)),
        delay_penalty=float(tcfg.get("delay_penalty", 0.0)),
    )
    loss = combine_loss(simple, pruned, ctc, batch_idx=batch_idx, tcfg=tcfg)
    return loss, simple, pruned, ctc


def train_step(model, opt, scheduler, scaler, batch, device, tcfg, batch_idx: int, *, epoch: int, amp_dtype, use_amp: bool):
    """One optimizer update. Returns ``(loss, simple, pruned, ctc, nframes)``, or None.

    Losses stay on the device. Every ``.item()``, ``.tolist()`` or Python
    ``if`` on a CUDA tensor makes the CPU wait for the GPU queue to drain, and
    the GPU then idles while Python launches the next kernels. Lengths and
    targets are read from the CPU batch for that reason.

    With an fp16 GradScaler a non-finite loss is not checked here: its
    gradients are non-finite, and ``scaler.step`` skips the update.
    """
    index = _keep_indices(batch["feat_lens"], batch["target_lens"])
    if index.numel() == 0:
        return None
    keys = ("feats", "feat_lens", "targets", "target_lens")
    cpu = {k: batch[k] for k in keys}
    if index.numel() != cpu["feat_lens"].numel():
        cpu = _take(cpu, index)
    feats = cpu["feats"].to(device, non_blocking=True)
    lens = cpu["feat_lens"].to(device, non_blocking=True)
    targets, target_lens = cpu["targets"], cpu["target_lens"]
    nframes = sum(encoder_frames(int(n)) for n in cpu["feat_lens"].tolist())
    if tcfg.get("specaug", True) and epoch >= int(tcfg.get("specaug_start_epoch", 1)):
        feats = spec_augment(feats, lens)

    def _fwd():
        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            return _forward_loss(model, feats, lens, targets, target_lens, tcfg, batch_idx)

    def _backward(loss) -> bool:
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            return True
        if not torch.isfinite(loss):
            return False
        loss.backward()
        return True

    try:
        loss, simple, pruned, ctc = _fwd()
        did_backward = _backward(loss)
    except Exception as exc:
        compiled = getattr(model.encoder, "_compiled_inner", None)
        if compiled is None or not _compile_failure(exc) or getattr(model, "_compile_ok", False):
            raise
        log.warning("compiled encoder failed (%s); falling back to eager", exc)
        model.encoder._compiled_inner = None
        opt.zero_grad(set_to_none=True)
        loss, simple, pruned, ctc = _fwd()
        did_backward = _backward(loss)
    else:
        if did_backward and getattr(model.encoder, "_compiled_inner", None) is not None:
            model._compile_ok = True
    if not did_backward:
        log.warning("epoch %d batch %d non-finite loss; skipping the update", epoch, batch_idx)
        opt.zero_grad(set_to_none=True)
        return None
    scheduler.step_batch(batch_idx)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    return loss.detach(), simple.detach(), pruned.detach(), ctc.detach(), nframes


def rnnt_decode_epoch(epoch: int, epochs: int, every: int) -> bool:
    """Transducer greedy walks one encoder frame at a time.

    ``every`` of 1 keeps that walk on every dev pass. A larger value scores
    those epochs with CTC greedy (one argmax) and runs the walk on epoch 1,
    the last epoch, and every ``every``th epoch.
    """
    every = int(every)
    if every <= 1:
        return True
    return epoch == 1 or epoch == int(epochs) or epoch % every == 0


def _min_recorded(history: list[dict], key: str, fallback: float = 1e9) -> float:
    scores = [fallback]
    for row in history:
        value = row.get(key)
        if isinstance(value, (int, float)) and value == value:
            scores.append(float(value))
    return min(scores)


@torch.no_grad()
def evaluate(model, loader, tokenizer, device, tcfg, batch_idx: int, *, decode_rnnt: bool = True) -> dict:
    model.eval()
    refs, hyps, ctc_hyps = [], [], []
    total, frames = 0.0, 0
    shown = []
    for batch in loader:
        index = _keep_indices(batch["feat_lens"], batch["target_lens"])
        if index.numel() == 0:
            continue
        if index.numel() != batch["feat_lens"].numel():
            batch = _take(batch, index)
        nframes = sum(encoder_frames(int(n)) for n in batch["feat_lens"].tolist())
        feats = batch["feats"].to(device, non_blocking=True)
        lens = batch["feat_lens"].to(device, non_blocking=True)
        targets, target_lens = batch["targets"], batch["target_lens"]
        frames += max(1, nframes)
        if decode_rnnt:
            loss, _, _, _ = _forward_loss(model, feats, lens, targets, target_lens, tcfg, batch_idx)
            total += float(loss.detach())
        encoder_out, encoder_out_lens = model.forward_encoder(feats, lens)
        if decode_rnnt:
            for ids in greedy_search_batch(model, encoder_out, encoder_out_lens):
                hyps.append(tokenizer.decode(ids, collapse=False))
        ctc_log = model.ctc_output(encoder_out)
        ctc_hyps.extend(greedy_decode(ctc_log, tokenizer, encoder_out_lens))
        refs.extend(batch["text"])
        if len(shown) < 3:
            chosen = hyps if decode_rnnt else ctc_hyps
            system = "rnnt" if decode_rnnt else "ctc"
            for ref, hyp in zip(batch["text"], chosen[-len(batch["text"]) :]):
                if len(shown) < 3:
                    shown.append({"ref": ref, "hyp": hyp, "system": system})
    blank = {"wer": float("nan"), "cer": float("nan")}
    ctc = score_texts(refs, ctc_hyps, fold_geez=False) if refs else blank
    if decode_rnnt and refs:
        rnnt = score_texts(refs, hyps, fold_geez=False)
        rnnt_wer, rnnt_cer = rnnt["wer"], rnnt["cer"]
        dev_loss = total / max(1, frames)
    else:
        rnnt_wer = rnnt_cer = dev_loss = None
    return {
        "wer": rnnt_wer,
        "cer": rnnt_cer,
        "ctc_wer": ctc["wer"],
        "ctc_cer": ctc["cer"],
        "ctc_substitutions": int(ctc.get("substitutions", 0)),
        "ctc_deletions": int(ctc.get("deletions", 0)),
        "ctc_insertions": int(ctc.get("insertions", 0)),
        "ctc_hits": int(ctc.get("hits", 0)),
        "rnnt_substitutions": int(rnnt.get("substitutions", 0)) if decode_rnnt and refs else None,
        "rnnt_deletions": int(rnnt.get("deletions", 0)) if decode_rnnt and refs else None,
        "rnnt_insertions": int(rnnt.get("insertions", 0)) if decode_rnnt and refs else None,
        "loss": dev_loss,
        "samples": shown,
        "decode_rnnt": decode_rnnt,
    }


def _compile_encoder(model, tcfg: dict) -> None:
    """Compile the training encoder body. Chunk size stays outside the graph.

    Off unless ``train.compile_encoder`` is set. On a Quadro T2000 a compiled
    step had about 20% fewer CUDA kernels and a slightly higher step time,
    and dynamo kept invalidating pieces of the graph (balancer, whitener, and
    the k2 swoosh kernel cannot sit inside one compiled region). Eval stays
    eager either way.
    """
    if not bool(tcfg.get("compile_encoder", False)):
        return
    encoder = getattr(model, "encoder", None)
    if encoder is None or not hasattr(encoder, "_forward_inner"):
        return
    if not torch.cuda.is_available():
        log.info("encoder compile skipped (no CUDA)")
        return
    try:
        limit = max(16, int(torch._dynamo.config.cache_size_limit))
        torch._dynamo.config.cache_size_limit = limit
        encoder._compiled_inner = torch.compile(encoder._forward_inner, dynamic=True)
        log.info("encoder compile on (dynamic, cache_size_limit=%d); eval stays eager", limit)
    except Exception as exc:
        encoder._compiled_inner = None
        log.warning("encoder compile unavailable (%s); training stays eager", exc)


def _compile_failure(exc: BaseException) -> bool:
    name = type(exc).__name__
    return "Dynamo" in name or "Inductor" in name or "BackendCompiler" in name


_save_requested = False


def _on_save_signal(signum, frame) -> None:
    """Record SIGUSR1. The training loop writes the file between steps."""
    global _save_requested
    _save_requested = True


def _to_cpu(obj):
    """Copy tensors to CPU. Leaves the live GPU parameters where they are."""
    if torch.is_tensor(obj):
        if obj.device.type == "cpu":
            return obj.detach().clone()
        return obj.detach().to("cpu")
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(v) for v in obj)
    return obj


def _save_cpu_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(_to_cpu(payload), tmp)
    tmp.replace(path)


def train(cfg: dict, *, init_from: str | None = None, resume: str | None = None) -> Path:
    load_icefall()
    from icefall.utils import create_grad_scaler

    set_seed(int(cfg.get("seed", 42)))
    exp_dir = ROOT / cfg["paths"]["exp_dir"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_json(exp_dir / "config.json", cfg)

    if init_from is None:
        init_from = cfg.get("paths", {}).get("init_from")
    if init_from:
        init_from = str(Path(init_from) if Path(init_from).is_absolute() else ROOT / init_from)
    if resume and init_from:
        raise ValueError("use either --resume or --init-from, not both")
    if init_from:
        copy_init_sidecars(Path(init_from), exp_dir)

    tokenizer = load_tokenizer(cfg, exp_dir)
    model = build_zipformer(cfg, tokenizer.vocab_size)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("parameters: %s (%.2fM) vocab=%d", f"{n_params:,}", n_params / 1e6, tokenizer.vocab_size)
    log.info("delay_penalty=%s", float(cfg.get("train", {}).get("delay_penalty", 0.0)))
    save_json(exp_dir / "model_info.json", {"parameters": n_params, "vocab_size": tokenizer.vocab_size})

    if init_from:
        ckpt = torch.load(init_from, map_location="cpu", weights_only=False)
        src = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(src)
        log.info("init-from: loaded weights from %s", init_from)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tcfg = cfg["train"]
    cmvn = load_cmvn(cfg, exp_dir, compute_if_missing=True)
    train_loader = _rebuild(make_loader("train", tokenizer, cfg, train=True, cmvn=cmvn), cfg, True)
    dev_loader = _rebuild(make_loader("dev", tokenizer, cfg, train=False, cmvn=cmvn), cfg, False)
    _drop_short_labels(train_loader.dataset, tokenizer, "train")
    _drop_short_labels(dev_loader.dataset, tokenizer, "dev")
    train_loader = _rebuild(train_loader, cfg, True)
    dev_loader = _rebuild(dev_loader, cfg, False)
    log.info("train clips=%d  steps/epoch≈%d", len(train_loader.dataset), len(train_loader))

    base_lr = float(tcfg["base_lr"])
    opt = scaled_adam(model, base_lr)
    scheduler = eden(opt, tcfg)
    use_amp, amp_dtype = _amp_setup(tcfg, device)
    # zipformer/train.py builds the scaler with init_scale=1.0 whenever autocast
    # is on. ScaledAdam does its own clipping, so the usual 65536 start is not
    # what that recipe uses. bf16 does not need a scaler; fp16 does.
    scaler = create_grad_scaler(
        device.type if device.type == "cuda" else "cpu",
        enabled=use_amp and amp_dtype == torch.float16,
        init_scale=1.0,
    )
    torch.backends.cudnn.enabled = bool(tcfg.get("cudnn", False))
    # set_seed turns benchmark on. Duration batching gives a new shape almost
    # every step, and benchmark re-tunes each one: 12.8 s/step against 1.8 s
    # on a 60 s batch (Quadro T2000).
    torch.backends.cudnn.benchmark = bool(tcfg.get("cudnn_benchmark", False))
    log.info(
        "amp enabled=%s dtype=%s cudnn=%s benchmark=%s",
        use_amp,
        amp_dtype,
        torch.backends.cudnn.enabled,
        torch.backends.cudnn.benchmark,
    )
    _compile_encoder(model, tcfg)

    epochs = int(tcfg.get("epochs", 30))
    patience = int(tcfg.get("early_stopping_patience", 8))
    rnnt_every = int(tcfg.get("rnnt_decode_every", 1))
    batch_seconds = float(tcfg.get("batch_seconds", 80))
    ref_duration = float(tcfg.get("ref_duration", 600))
    best_wer = 1e9
    best_ctc_wer = 1e9
    bad = 0
    batch_idx = 0
    start_epoch = 1
    history: list[dict] = []
    hist_path = exp_dir / "history.json"

    if resume:
        resume_path = Path(resume) if Path(resume).is_absolute() else ROOT / resume
        blob = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["optimizer"])
        scheduler.load_state_dict(blob["scheduler"])
        if scaler.is_enabled() and blob.get("scaler"):
            scaler.load_state_dict(blob["scaler"])
        done = int(blob.get("epoch") or 0)
        start_epoch = done + 1
        batch_idx = int(blob.get("batch_idx") or 0)
        best_wer = float(blob.get("best_wer") or 1e9)
        bad = int(blob.get("bad") or 0)
        history = _merge_history(_load_history_file(hist_path), blob.get("history") or [])
        history = [h for h in history if int(h.get("epoch", 0)) <= done]
        stored_ctc = blob.get("best_ctc_wer")
        best_ctc_wer = _min_recorded(
            history,
            "dev_ctc_wer",
            float(stored_ctc) if isinstance(stored_ctc, (int, float)) else 1e9,
        )
        # Older checkpoints counted `bad` on the transducer number. The
        # every-epoch check is now CTC, so that count starts over once.
        if not isinstance(stored_ctc, (int, float)):
            bad = 0
        sampler = getattr(train_loader, "batch_sampler", None)
        if sampler is not None and "sampler_seed" in blob:
            sampler.seed = int(blob["sampler_seed"])
        _restore_rng(blob.get("rng"))
        log.info(
            "resume %s: next epoch %d/%d step=%d best_rnnt_wer=%.2f best_ctc_wer=%.2f rnnt_decode_every=%d",
            resume_path,
            start_epoch,
            epochs,
            batch_idx,
            best_wer * 100,
            best_ctc_wer * 100,
            rnnt_every,
        )

    signal.signal(signal.SIGUSR1, _on_save_signal)
    save_flag = exp_dir / "SAVE_NOW"

    waveform_augment = train_loader.dataset.augment
    augment_start = int(tcfg.get("augment_start_epoch", 1))
    if waveform_augment is not None and start_epoch < augment_start:
        train_loader.dataset.augment = None
        log.info("waveform augmentation starts at epoch %d", augment_start)

    for epoch in range(start_epoch, epochs + 1):
        train_loader = _start_waveform_augment(train_loader, waveform_augment, epoch, augment_start, cfg)
        model.train()
        scheduler.step_epoch(epoch - 1)
        t0 = time.time()
        # Summed on the device and read only when logging.
        running = torch.zeros((), device=device)
        nonfinite = torch.zeros((), device=device)
        seen_frames = 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, 1):
            set_batch_count(model, adjusted_batch_count(batch_idx, batch_seconds, ref_duration))
            batch_idx += 1
            out = train_step(
                model, opt, scheduler, scaler, batch, device, tcfg, batch_idx,
                epoch=epoch, amp_dtype=amp_dtype, use_amp=use_amp,
            )
            if out is None:
                continue
            loss, simple, pruned, ctc, nframes = out
            finite = torch.isfinite(loss)
            running += torch.where(finite, loss, torch.zeros_like(loss))
            nonfinite += (~finite).to(nonfinite.dtype)
            seen_frames += max(1, nframes)
            if step % 50 == 0:
                skipped = int(nonfinite)
                if skipped:
                    log.warning("epoch %d: %d non-finite losses so far; the scaler skipped those updates", epoch, skipped)
                simple_s, pruned_s, ctc_s = loss_scales(batch_idx, tcfg)
                mem = ""
                if device.type == "cuda":
                    gb = 1024**3
                    mem = "  mem reserved %.1f GB" % (torch.cuda.memory_reserved(device) / gb)
                log.info(
                    "epoch %d step %d loss %.4f simple %.3f pruned %.3f ctc %.3f "
                    "scales %.2f/%.2f/%.2f lr %.2e%s",
                    epoch,
                    step,
                    float(running) / seen_frames,
                    float(simple) / max(1, nframes),
                    float(pruned) / max(1, nframes),
                    float(ctc) / max(1, nframes),
                    simple_s,
                    pruned_s,
                    ctc_s,
                    opt.param_groups[0]["lr"],
                    mem,
                )
            global _save_requested
            if _save_requested or save_flag.exists():
                _save_requested = False
                save_flag.unlink(missing_ok=True)
                emergency = exp_dir / "emergency.pt"
                sampler = getattr(train_loader, "batch_sampler", None)
                _save_cpu_checkpoint(
                    emergency,
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict() if scaler.is_enabled() else {},
                        "config": cfg,
                        "vocab_size": tokenizer.vocab_size,
                        "batch_idx": batch_idx,
                        "best_wer": best_wer,
                        "best_ctc_wer": best_ctc_wer,
                        "bad": bad,
                        "sampler_seed": getattr(sampler, "seed", 0),
                        "history": history,
                        "rng": _capture_rng(),
                    },
                )
                log.info("wrote %s at epoch %d step %d (GPU weights left in place)", emergency, epoch, step)
        decode_rnnt = rnnt_decode_epoch(epoch, epochs, rnnt_every)
        if decode_rnnt:
            log.info("dev: transducer greedy")
        else:
            log.info("dev: CTC greedy")
        dev = evaluate(
            model, dev_loader, tokenizer, device, tcfg, batch_idx, decode_rnnt=decode_rnnt
        )
        rec = {
            "epoch": epoch,
            "train_loss": float(running) / max(1, seen_frames),
            "dev_wer": dev["wer"],
            "dev_cer": dev["cer"],
            "dev_ctc_wer": dev["ctc_wer"],
            "dev_ctc_cer": dev["ctc_cer"],
            "dev_ctc_substitutions": dev["ctc_substitutions"],
            "dev_ctc_deletions": dev["ctc_deletions"],
            "dev_ctc_insertions": dev["ctc_insertions"],
            "dev_ctc_hits": dev["ctc_hits"],
            "dev_rnnt_substitutions": dev["rnnt_substitutions"],
            "dev_rnnt_deletions": dev["rnnt_deletions"],
            "dev_rnnt_insertions": dev["rnnt_insertions"],
            "delay_penalty": float(tcfg.get("delay_penalty", 0.0)),
            "dev_loss": dev["loss"],
            "seconds": round(time.time() - t0, 1),
            "lr": opt.param_groups[0]["lr"],
            "batch_idx": batch_idx,
        }
        history = [h for h in history if int(h.get("epoch", 0)) < epoch]
        history.append(rec)
        history = _save_history(hist_path, history)
        if dev["wer"] is None:
            log.info(
                "epoch %d  train_loss=%.4f  dev_ctc_wer=%.2f  dev_ctc_cer=%.2f  "
                "ctc S/D/I %d/%d/%d  delay %.4f  (%.0fs)",
                epoch,
                rec["train_loss"],
                rec["dev_ctc_wer"] * 100,
                rec["dev_ctc_cer"] * 100,
                rec["dev_ctc_substitutions"],
                rec["dev_ctc_deletions"],
                rec["dev_ctc_insertions"],
                rec["delay_penalty"],
                rec["seconds"],
            )
        else:
            log.info(
                "epoch %d  train_loss=%.4f  dev_rnnt_wer=%.2f  dev_rnnt_cer=%.2f  "
                "dev_ctc_wer=%.2f  dev_ctc_cer=%.2f  ctc S/D/I %d/%d/%d  delay %.4f  (%.0fs)",
                epoch,
                rec["train_loss"],
                rec["dev_wer"] * 100,
                rec["dev_cer"] * 100,
                rec["dev_ctc_wer"] * 100,
                rec["dev_ctc_cer"] * 100,
                rec["dev_ctc_substitutions"],
                rec["dev_ctc_deletions"],
                rec["dev_ctc_insertions"],
                rec["delay_penalty"],
                rec["seconds"],
            )
        for i, sample in enumerate(dev["samples"]):
            log.info("  sample %d ref: %s", i, (sample["ref"] or "")[:160])
            log.info(
                "  sample %d %s hyp: %s",
                i,
                sample.get("system", "hyp"),
                (sample["hyp"] or "")[:160],
            )
        is_best = False
        if dev["ctc_wer"] < best_ctc_wer:
            best_ctc_wer = dev["ctc_wer"]
            bad = 0
            is_best = True
            log.info("  new best CTC WER %.2f", best_ctc_wer * 100)
        else:
            bad += 1
        if dev["wer"] is not None and dev["wer"] < best_wer:
            best_wer = dev["wer"]
            is_best = True
            log.info("  new best transducer WER %.2f", best_wer * 100)
        sampler = getattr(train_loader, "batch_sampler", None)
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else {},
            "config": cfg,
            "vocab_size": tokenizer.vocab_size,
            "dev": dev,
            "batch_idx": batch_idx,
            "best_wer": best_wer,
            "best_ctc_wer": best_ctc_wer,
            "bad": bad,
            "sampler_seed": getattr(sampler, "seed", 0),
            "history": history,
            "rng": _capture_rng(),
        }
        torch.save(ckpt, exp_dir / "last.pt")
        if is_best:
            torch.save(ckpt, exp_dir / "best.pt")
        if bad >= patience:
            log.info("early stopping at epoch %d", epoch)
            break
    return exp_dir / "best.pt"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--init-from", type=str, default=None)
    p.add_argument("--resume", nargs="?", const="", default=None)
    args = p.parse_args()
    cfg = load_yaml(args.config)
    resume = args.resume
    if resume is not None and resume == "":
        resume = str(ROOT / cfg["paths"]["exp_dir"] / "last.pt")
    train(cfg, init_from=args.init_from, resume=resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
