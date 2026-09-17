#!/usr/bin/env python3
"""Train a Conformer CTC model.

    python train.py --config configs/20m_char.yaml
    python train.py --config configs/20m_char.yaml --overfit-batch
    python train.py --config configs/20m_char_4layer.yaml --resume
    python train.py --config configs/20m_char_4layer.yaml --resume exp/20m_char_4layer/best.pt
"""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import time
from pathlib import Path

# Every duration batch has a new shape. Fixed-size cache blocks fragment and the
# reserved pool creeps toward the card limit; growable segments reuse one range.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import DataLoader

from data.dataset import AsrDataset, DurationBatchSampler, collate
from data.prepare import load_split_jsonl, metadata_dir
from decoding.greedy import greedy_decode
from evaluation.metrics import score_texts
from models.ctc import build_model, param_count
from models.features import GlobalCMVN, SpecAugment, compute_cmvn
from tokenizer.char import CharTokenizer
from tokenizer.phoneme import PhonemeTokenizer
from tokenizer.piece import PieceTokenizer
from tokenizer.train_bpe import SpmTokenizer
from utils import ROOT, load_json, load_yaml, save_json, set_seed, setup_logging

log = setup_logging()


def _resolve_ckpt(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def copy_init_sidecars(ckpt_path: Path, exp_dir: Path) -> list[str]:
    """Reuse the donor run's tokenizer and CMVN so the CTC head stays aligned.

    Overwrites what is in ``exp_dir``: files left by an earlier cold start
    would give the CTC head a different vocabulary and the features
    different statistics from the ones the donor learned.
    """
    src_dir = ckpt_path.resolve().parent
    copied: list[str] = []
    for name in ("tokenizer.json", "cmvn.pt", "spm.model", "spm.vocab"):
        src, dst = src_dir / name, exp_dir / name
        if not src.exists() or (dst.exists() and src.read_bytes() == dst.read_bytes()):
            continue
        if dst.exists():
            log.warning("init-from: replacing %s with the donor's copy", dst)
        shutil.copy2(src, dst)
        copied.append(name)
        log.info("init-from: copied %s from %s", name, src_dir)
    return copied


def transfer_depthwise(src: dict, cur: dict) -> tuple[list[str], list[str]]:
    """Map a single-kernel ``conv.dw`` into multi-kernel branches.

    Exact kernel-size matches copy as-is (the 31-tap branch from the 20 M
    baseline). Smaller odd kernels take the centre of the donor filter.
    Returns ``(dest_keys, donor_keys)``.
    """
    copied: list[str] = []
    used: list[str] = []
    for k, v in src.items():
        if not k.endswith(".conv.dw.weight"):
            continue
        stem = k[: -len("dw.weight")]
        src_k = int(v.shape[-1])
        bias_k = k[: -len("weight")] + "bias"
        src_bias = src.get(bias_k)
        for dest_k, dest_v in cur.items():
            if dest_k == k or not dest_k.startswith(stem + "dw.") or not dest_k.endswith(".weight"):
                continue
            if dest_v.shape[:-1] != v.shape[:-1]:
                continue
            dest_ksize = int(dest_v.shape[-1])
            if dest_ksize == src_k:
                cur[dest_k] = v.clone()
            elif dest_ksize < src_k and (src_k - dest_ksize) % 2 == 0:
                off = (src_k - dest_ksize) // 2
                cur[dest_k] = v[..., off : off + dest_ksize].clone()
            else:
                continue
            copied.append(dest_k)
            used.append(k)
            dest_b = dest_k[: -len("weight")] + "bias"
            if src_bias is not None and dest_b in cur and cur[dest_b].shape == src_bias.shape:
                cur[dest_b] = src_bias.clone()
                copied.append(dest_b)
                used.append(bias_k)
    m_copied, m_used = _merge_depthwise_branches(src, cur)
    return copied + m_copied, used + m_used


def _place_depthwise_taps(w: torch.Tensor, width: int) -> torch.Tensor:
    """Center a depthwise kernel in ``width`` taps, cropping or zero-padding."""
    kw = int(w.shape[-1])
    if kw == width:
        return w
    if kw < width:
        out = w.new_zeros(*w.shape[:-1], width)
        off = (width - kw) // 2
        out[..., off : off + kw] = w
        return out
    off = (kw - width) // 2
    return w[..., off : off + width]


def _stretch_depthwise_taps(w: torch.Tensor) -> torch.Tensor:
    """Resample taps from a 40 ms grid onto 20 ms. ``K`` taps become ``2K-1``.

    Even positions keep the donor taps. Odd positions are the midpoint of the
    taps on either side, which is the same impulse response sampled twice as
    often.
    """
    out = w.new_zeros(*w.shape[:-1], 2 * int(w.shape[-1]) - 1)
    out[..., 0::2] = w
    out[..., 1::2] = 0.5 * (w[..., :-1] + w[..., 1:])
    return out


def _merge_depthwise_branches(src: dict, cur: dict) -> tuple[list[str], list[str]]:
    """Fold multi-kernel ``conv.dw.{i}`` donors into one ``conv.dw`` filter.

    The donor averages its branches. Each branch is centre-placed on the
    widest branch (zero-padded, or cropped if the destination is narrower)
    and the mean is taken. A destination of ``2 * widest - 1`` taps is the
    20 ms copy of a 40 ms filter: the averaged impulse is stretched so a
    31-tap branch fills 61 taps instead of sitting in the middle with zeros
    on either side.
    """
    copied: list[str] = []
    used: list[str] = []
    for dest_k, dest_v in cur.items():
        if not dest_k.endswith(".conv.dw.weight") or dest_k in src:
            continue
        stem = dest_k[: -len("weight")]
        branches = sorted(k for k in src if k.startswith(stem) and k.endswith(".weight") and k[len(stem) :].split(".")[0].isdigit())
        if not branches or any(src[k].shape[:-1] != dest_v.shape[:-1] for k in branches):
            continue
        width = int(dest_v.shape[-1])
        max_kw = max(int(src[k].shape[-1]) for k in branches)
        placed = dest_v.new_zeros(*dest_v.shape[:-1], max_kw)
        for k in branches:
            placed = placed + _place_depthwise_taps(src[k], max_kw)
        placed = placed / len(branches)
        if width == 2 * max_kw - 1:
            cur[dest_k] = _stretch_depthwise_taps(placed)
        else:
            cur[dest_k] = _place_depthwise_taps(placed, width)
        copied.append(dest_k)
        used.extend(branches)
        biases = [k[: -len("weight")] + "bias" for k in branches]
        dest_b = stem + "bias"
        if dest_b in cur and all(b in src for b in biases):
            cur[dest_b] = sum(src[b] for b in biases) / len(biases)
            copied.append(dest_b)
            used.extend(biases)
    return copied, used


def transfer_mha_to_relpos(src: dict, cur: dict) -> tuple[list[str], list[str]]:
    """Copy fused ``nn.MultiheadAttention`` weights into ``RelSelfAttention``.

    PyTorch packs Q/K/V as ``in_proj_{weight,bias}`` with the same layout as
    ``nn.Linear(d, 3d)``. Content attention therefore transfers; relative
    ``pos_proj`` / ``u`` / ``v`` stay at init because the donor has no analogue.
    Returns ``(dest_keys, donor_keys)``.
    """
    copied: list[str] = []
    used: list[str] = []
    pairs = (
        (".attn.attn.in_proj_weight", ".attn.qkv.weight"),
        (".attn.attn.in_proj_bias", ".attn.qkv.bias"),
        (".attn.attn.out_proj.weight", ".attn.out.weight"),
        (".attn.attn.out_proj.bias", ".attn.out.bias"),
    )
    for src_suffix, dest_suffix in pairs:
        for src_k, v in src.items():
            if not src_k.endswith(src_suffix):
                continue
            dest_k = src_k[: -len(src_suffix)] + dest_suffix
            if dest_k not in cur or cur[dest_k].shape != v.shape:
                continue
            cur[dest_k] = v.clone()
            copied.append(dest_k)
            used.append(src_k)
    return copied, used


def load_tokenizer(cfg: dict, exp_dir: Path):
    kind = cfg["tokenizer"]["type"]
    tok_path = exp_dir / "tokenizer.json"
    spm_path = exp_dir / "spm.model"
    if kind == "char":
        if tok_path.exists():
            return CharTokenizer.load(tok_path)
        rows = load_split_jsonl("train", metadata_dir(cfg))
        tok = CharTokenizer.from_texts(r["text"] for r in rows)
        tok.save(tok_path)
        return tok
    if kind == "phoneme":
        if tok_path.exists():
            return PhonemeTokenizer.load(tok_path)
        tok = PhonemeTokenizer.default()
        tok.save(tok_path)
        return tok
    if kind in ("piece", "ideal_bpe"):
        src = Path(cfg["tokenizer"]["path"]) if cfg["tokenizer"].get("path") else tok_path
        if not src.exists():
            raise FileNotFoundError(
                f"piece tokenizer missing at {src}; run python -m tokenizer.build_ideal_bpe"
            )
        tok = PieceTokenizer.load(src)
        if src.resolve() != tok_path.resolve():
            tok.save(tok_path)
        return tok
    if kind in ("bpe", "unigram"):
        if not spm_path.exists():
            from tokenizer.train_bpe import train_spm

            rows = load_split_jsonl("train", metadata_dir(cfg))
            train_spm(
                [r["text"] for r in rows],
                vocab_size=int(cfg["tokenizer"]["vocab_size"]),
                model_type="unigram" if kind == "unigram" else "bpe",
                out_prefix=exp_dir / "spm",
            )
        return SpmTokenizer(spm_path)
    raise ValueError(f"unknown tokenizer {kind}")


def cosine_with_warmup(step: int, *, warmup: int, total: int, base_lr: float, min_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * min(1.0, progress)))


def load_cmvn(cfg: dict, exp_dir: Path, *, compute_if_missing: bool = False) -> GlobalCMVN | None:
    """Load the CMVN stats the model was trained with.

    Eval must use the saved ``cmvn.pt``. Recomputing on the fly would shift the
    feature scale and silently destroy greedy WER.
    """
    if not cfg.get("train", {}).get("cmvn", True):
        return None
    cmvn_path = exp_dir / "cmvn.pt"
    if cmvn_path.exists():
        cmvn = GlobalCMVN.load(cmvn_path)
        log.info("loaded CMVN %s", cmvn_path)
        return cmvn
    if not compute_if_missing:
        raise FileNotFoundError(
            f"CMVN is enabled but {cmvn_path} is missing. Train first so eval uses the same stats."
        )
    log.info("computing CMVN from a train subset")
    cmvn = compute_cmvn(metadata_dir(cfg) / "train.jsonl", root=ROOT)
    cmvn.save(cmvn_path)
    log.info("wrote %s", cmvn_path)
    return cmvn


def _capture_rng() -> dict:
    state: dict = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    return state


def _restore_rng(state: dict | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        rng = state["torch"]
        torch.set_rng_state(rng.cpu() if torch.is_tensor(rng) else rng)
    if "cuda" in state and torch.cuda.is_available():
        try:
            cuda_states = state["cuda"]
            for i, s in enumerate(cuda_states):
                if torch.is_tensor(s):
                    s = s.to(dtype=torch.uint8).cpu()
                torch.cuda.set_rng_state(s, i)
        except Exception:
            pass  # different GPU count or dtype; harmless to skip
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])


def _patience_from_history(history: list[dict]) -> tuple[float, int]:
    if not history:
        return 1e9, 0
    best = 1e9
    best_epoch = 0
    for rec in history:
        wer = float(rec["dev_wer"])
        if wer < best:
            best = wer
            best_epoch = int(rec["epoch"])
    bad = int(history[-1]["epoch"]) - best_epoch
    return best, bad


def _merge_history(*parts: list) -> list[dict]:
    """Union of history records keyed by epoch. Later parts overwrite the same epoch."""
    by_epoch: dict[int, dict] = {}
    for part in parts:
        for rec in part or []:
            try:
                epoch = int(rec.get("epoch", -1))
            except (TypeError, ValueError):
                continue
            if epoch >= 0:
                by_epoch[epoch] = rec
    return [by_epoch[e] for e in sorted(by_epoch)]


def _load_history_file(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = load_json(path)
    except Exception:
        return []
    return list(data) if isinstance(data, list) else []


def _save_history(path: Path, history: list[dict], *, union_disk: bool = False) -> list[dict]:
    if union_disk:
        history = _merge_history(_load_history_file(path), history)
    save_json(path, history)
    return history


def _amp_dtype(requested: str, *, cuda: bool, bf16_tensor_cores: bool) -> tuple[bool, torch.dtype]:
    """Pick the autocast dtype. BF16 only when the GPU has tensor cores for it.

    ``is_bf16_supported()`` is true on a T4 because CUDA can emulate the
    type. On that card a 50M training step was 2.6× slower in bf16 than fp16.
    """
    requested = requested.lower()
    if not cuda or requested not in ("bf16", "fp16"):
        return False, torch.float32
    if requested == "bf16" and not bf16_tensor_cores:
        requested = "fp16"
    dtype = torch.bfloat16 if requested == "bf16" else torch.float16
    return True, dtype


def _bf16_tensor_cores(device: torch.device) -> bool:
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        major, _ = torch.cuda.get_device_capability(device)
        return major >= 8


def _amp_setup(tcfg: dict, device: torch.device) -> tuple[bool, torch.dtype]:
    requested = str(tcfg.get("precision", "bf16"))
    cuda = device.type == "cuda"
    enabled, dtype = _amp_dtype(requested, cuda=cuda, bf16_tensor_cores=cuda and _bf16_tensor_cores(device))
    if requested.lower() == "bf16" and dtype == torch.float16:
        log.warning(
            "precision=bf16 has no tensor cores on %s; using fp16.",
            torch.cuda.get_device_name(device),
        )
    return enabled, dtype


def _make_adamw(params, tcfg: dict, device: torch.device, *, lr: float | None = None) -> torch.optim.Optimizer:
    kw = dict(
        lr=float(tcfg["lr"] if lr is None else lr),
        weight_decay=float(tcfg.get("weight_decay", 0.01)) if lr is None else 0.0,
        betas=(0.9, 0.98),
    )
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(params, fused=True, **kw)
        except (TypeError, RuntimeError) as exc:
            log.warning("fused AdamW unavailable (%s); using default", exc)
    return torch.optim.AdamW(params, **kw)


def make_loader(split: str, tokenizer, cfg: dict, train: bool, cmvn=None) -> DataLoader:
    tcfg = cfg["train"]
    from data.augment import AugmentConfig, WaveformAugment

    aug = None
    if train:
        acfg = AugmentConfig.from_train_cfg(tcfg)
        if acfg is not None:
            aug = WaveformAugment(acfg)
            log.info(
                "augment: gain=%s gain_rel=%s@%.2f noise_p=%.2f rir_p=%.2f codec_p=%.2f (%s) noise_files=%d rir_files=%d",
                acfg.gain_db,
                acfg.gain_rel_db,
                acfg.gain_rel_prob,
                acfg.noise_prob,
                acfg.rir_prob,
                acfg.codec_prob,
                acfg.codec,
                len(aug.noise_files),
                len(aug.rir_files),
            )
    ds = AsrDataset(
        metadata_dir(cfg) / f"{split}.jsonl",
        tokenizer,
        train=train,
        speed_perturb_factors=tuple(tcfg.get("speed_perturb", [1.0])) if train else (1.0,),
        trim_silence=bool(tcfg.get("trim_silence", False)),
        cmvn=cmvn,
        max_utts=int(tcfg["max_train_utts"]) if train and tcfg.get("max_train_utts") else None,
        subset_seed=int(cfg.get("seed", 42)),
        augment=aug,
        max_duration=None if tcfg.get("max_duration", 30.0) is None else float(tcfg.get("max_duration", 30.0)),
        short_crop=tcfg.get("short_crop") if train else None,
    )
    if ds.dropped_long:
        log.info("%s: dropped %d clips longer than max_duration=%ss", split, ds.dropped_long, ds.max_duration)
    if train and tcfg.get("short_crop"):
        log.info(
            "%s: short crops p=%.2f, word timings on %d/%d clips",
            split,
            float(tcfg["short_crop"].get("prob", 0.0)),
            ds.crop_ready,
            len(ds),
        )
    n_workers = int(tcfg.get("num_workers", 4))
    loader_kw: dict = {"collate_fn": collate, "num_workers": n_workers, "pin_memory": True}
    if n_workers > 0:
        loader_kw["persistent_workers"] = True
        loader_kw["prefetch_factor"] = int(tcfg.get("prefetch_factor", 4))
    if train:
        sampler = DurationBatchSampler(ds, max_seconds=float(tcfg.get("batch_seconds", 80)), shuffle=True)
        return DataLoader(ds, batch_sampler=sampler, **loader_kw)
    return DataLoader(
        ds,
        batch_size=int(tcfg.get("eval_batch_size", 8)),
        shuffle=False,
        **loader_kw,
    )


@torch.no_grad()
def evaluate_model(model, loader, tokenizer, device, specaug=None) -> dict:
    model.eval()
    refs, hyps = [], []
    total_loss, n = 0.0, 0
    for batch in loader:
        feats = batch["feats"].to(device, non_blocking=True)
        lens = batch["feat_lens"].to(device, non_blocking=True)
        logits, logit_lens = model(feats, lens)
        loss = model.ctc_loss(
            logits,
            logit_lens,
            batch["targets"].to(device, non_blocking=True),
            batch["target_lens"].to(device, non_blocking=True),
        )
        total_loss += float(loss) * feats.size(0)
        n += feats.size(0)
        hyps.extend(greedy_decode(logits, tokenizer, logit_lens))
        refs.extend(batch["text_raw"])
    scores = score_texts(refs, hyps)
    scores["loss"] = total_loss / max(1, n)
    scores["samples"] = [{"ref": r, "hyp": h} for r, h in zip(refs[:4], hyps[:4])]
    return scores


def _overfit_batch(model, train_loader, tokenizer, device, tcfg: dict, exp_dir: Path) -> Path:
    """Repeat a tiny frozen set until CTC can spell it. No SpecAugment, dropout, or warmup."""
    batch = next(iter(train_loader))
    n_keep = int(tcfg.get("overfit_utts", 2))
    n_keep = min(n_keep, batch["feats"].size(0))
    lens = batch["feat_lens"][:n_keep]
    tlen = batch["target_lens"][:n_keep]
    feats = batch["feats"][:n_keep, : int(lens.max())].to(device)
    lens = lens.to(device)
    targets = batch["targets"][:n_keep, : int(tlen.max())].to(device)
    target_lens = tlen.to(device)
    refs = list(batch["text"][:n_keep])
    n_steps = int(tcfg.get("overfit_steps", 800))
    lr = float(tcfg.get("overfit_lr", 1.0e-3))
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
    log.info("overfit-batch: %d utts, %d steps, lr=%.1e (no specaug/dropout)", feats.size(0), n_steps, lr)

    opt = _make_adamw(model.parameters(), tcfg, device, lr=lr)
    use_amp, amp_dtype = _amp_setup(tcfg, device)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp and amp_dtype == torch.float16)
    clip = float(tcfg.get("grad_clip", 5.0))

    for step in range(1, n_steps + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            logits, logit_lens = model(feats, lens)
            loss = model.ctc_loss(logits, logit_lens, targets, target_lens)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(opt)
        scaler.update()
        if step == 1 or step % 25 == 0 or step == n_steps:
            model.eval()
            with torch.inference_mode():
                logits, logit_lens = model(feats, lens)
                hyps = greedy_decode(logits, tokenizer, logit_lens)
            cer = score_texts(refs, hyps, fold_geez=False)["cer"]
            log.info("overfit step %d  loss=%.4f  batch_cer=%.2f", step, float(loss.detach()), cer * 100)
            log.info("  ref: %s", (refs[0] or "")[:160])
            log.info("  hyp: %s", (hyps[0] or "")[:160])

    torch.save({"model": model.state_dict(), "config": None}, exp_dir / "overfit.pt")
    return exp_dir / "overfit.pt"


def train(
    cfg: dict,
    *,
    overfit_batch: bool = False,
    init_from: str | None = None,
    resume: str | None = None,
) -> Path:
    set_seed(int(cfg.get("seed", 42)))
    exp_dir = ROOT / cfg["paths"]["exp_dir"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_json(exp_dir / "config.json", cfg)

    if init_from is None:
        init_from = cfg.get("paths", {}).get("init_from")
    if init_from:
        init_from = str(_resolve_ckpt(init_from))

    if resume and init_from:
        raise ValueError("use either --resume (continue a run) or --init-from (weights only), not both")

    if init_from is not None:
        copy_init_sidecars(Path(init_from), exp_dir)

    tokenizer = load_tokenizer(cfg, exp_dir)
    model = build_model(cfg, tokenizer.vocab_size)
    n_params = param_count(model)
    log.info("parameters: %s (%.2fM) vocab=%d", f"{n_params:,}", n_params / 1e6, tokenizer.vocab_size)
    enc = cfg.get("model", {})
    if int(enc.get("subsample_factor", 4)) >= 8 and cfg.get("tokenizer", {}).get("type") == "char":
        log.warning(
            "8x subsampling + character CTC often violates T >= U on short clips "
            "(Fast Conformer switched to BPE for this reason). Prefer unigram/BPE."
        )
    save_json(exp_dir / "model_info.json", {"parameters": n_params, "vocab_size": tokenizer.vocab_size})

    if init_from is not None:
        ckpt = torch.load(init_from, map_location="cpu", weights_only=False)
        src = ckpt["model"] if "model" in ckpt else ckpt
        cur = model.state_dict()
        loaded, skipped = [], []
        for k, v in src.items():
            if k in cur and cur[k].shape == v.shape:
                cur[k] = v
                loaded.append(k)
            else:
                skipped.append(k)
        dw_copied, dw_used = transfer_depthwise(src, cur)
        attn_copied, attn_used = transfer_mha_to_relpos(src, cur)
        model.load_state_dict(cur)
        log.info("init-from: loaded %d/%d params from %s", len(loaded), len(src), init_from)
        if dw_copied:
            log.info("init-from: mapped %d depthwise tensors into multi-kernel branches", len(dw_copied))
        if attn_copied:
            log.info("init-from: mapped %d MultiheadAttention tensors into relpos qkv/out", len(attn_copied))
        leftover = [k for k in skipped if k not in set(dw_used) | set(attn_used)]
        if leftover:
            log.info("init-from: skipped %d params (shape mismatch): %s", len(leftover), ", ".join(leftover[:10]))
        elif skipped:
            log.info("init-from: remapped all shape-mismatched donor tensors")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tcfg = cfg["train"]
    cmvn = load_cmvn(cfg, exp_dir, compute_if_missing=True)

    if overfit_batch:
        freeze_loader = make_loader("train", tokenizer, cfg, train=False, cmvn=cmvn)
        return _overfit_batch(model, freeze_loader, tokenizer, device, tcfg, exp_dir)

    train_loader = make_loader("train", tokenizer, cfg, train=True, cmvn=cmvn)
    dev_loader = make_loader("dev", tokenizer, cfg, train=False, cmvn=cmvn)
    log.info("train clips=%d  steps/epoch≈%d", len(train_loader.dataset), len(train_loader))
    specaug = SpecAugment() if tcfg.get("specaug", True) else None
    specaug_start = int(tcfg.get("specaug_start_epoch", 1))
    opt = _make_adamw(model.parameters(), tcfg, device)
    use_amp, amp_dtype = _amp_setup(tcfg, device)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp and amp_dtype == torch.float16)
    # cuDNN sets up its convolution algorithm for every new input shape. With
    # duration batching nearly every step is a new shape; on a T4 in fp16 that
    # setup cost a 50M step 2.1 s against 0.45 s with PyTorch's own kernels.
    torch.backends.cudnn.enabled = bool(tcfg.get("cudnn", True))
    log.info(
        "amp enabled=%s dtype=%s fused_adam=%s cudnn=%s",
        use_amp,
        amp_dtype,
        bool(opt.defaults.get("fused", False)),
        torch.backends.cudnn.enabled,
    )

    epochs = int(tcfg.get("epochs", 40))
    warmup = int(tcfg.get("warmup_steps", 2000))
    steps_per_epoch = max(1, len(train_loader))
    total_steps = epochs * steps_per_epoch
    clip = float(tcfg.get("grad_clip", 5.0))
    patience = int(tcfg.get("early_stopping_patience", 8))
    accum = int(tcfg.get("grad_accum", 1))
    save_epoch_ckpts = bool(tcfg.get("save_epoch_ckpts", False))

    best_wer = 1e9
    bad = 0
    global_step = 0
    start_epoch = 1
    history: list[dict] = []
    hist_path = exp_dir / "history.json"
    union_hist = resume is not None

    if resume:
        resume_path = Path(resume)
        if not resume_path.is_absolute():
            resume_path = ROOT / resume_path
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume checkpoint not found: {resume_path}")
        blob = torch.load(resume_path, map_location=device, weights_only=False)
        if "vocab_size" in blob and int(blob["vocab_size"]) != tokenizer.vocab_size:
            raise ValueError(
                f"tokenizer vocab {tokenizer.vocab_size} != checkpoint vocab {blob['vocab_size']}"
            )
        model.load_state_dict(blob["model"])
        if "optimizer" in blob:
            try:
                opt.load_state_dict(blob["optimizer"])
            except Exception as exc:
                log.warning("resume: optimizer state not loaded (%s); Adam moments reset", exc)
        else:
            log.warning("resume: no optimizer state in %s; Adam moments reset", resume_path)
        if "scaler" in blob and scaler.is_enabled():
            src = blob.get("scaler") or {}
            if isinstance(src, dict) and src.get("enabled") is False:
                log.warning("resume: skipping scaler (checkpoint scaler disabled; fp16 needs a fresh one)")
            else:
                try:
                    scaler.load_state_dict(src)
                except Exception as exc:
                    log.warning("resume: scaler state not loaded (%s)", exc)
                # Below 1 the scaler is shrinking gradients, not protecting them,
                # and from 0 it can never grow back.
                if scaler.get_scale() < 1.0:
                    log.warning("resume: loss scale %.3g collapsed; restarting at 65536", scaler.get_scale())
                    scaler.update(new_scale=65536.0)
        done_epoch = int(blob.get("epoch") or 0)
        start_epoch = done_epoch + 1
        def _upto(recs: list) -> list[dict]:
            out = []
            for rec in recs or []:
                try:
                    if int(rec.get("epoch", -1)) <= done_epoch:
                        out.append(rec)
                except (TypeError, ValueError):
                    continue
            return out

        history = _merge_history(_upto(_load_history_file(hist_path)), _upto(blob.get("history") or []))
        log.info("resume: %d history epochs (union of disk + checkpoint)", len(history))
        if "global_step" in blob:
            global_step = int(blob["global_step"])
        else:
            global_step = done_epoch * steps_per_epoch
            log.warning("resume: global_step missing, approximating as %d", global_step)
        if "total_steps" in blob:
            total_steps = int(blob["total_steps"])
        else:
            old_epochs = int((blob.get("config") or {}).get("train", {}).get("epochs") or epochs)
            total_steps = old_epochs * steps_per_epoch
        if "best_wer" in blob:
            best_wer = float(blob["best_wer"])
            bad = int(blob.get("bad") or 0)
        else:
            # Old train.py last.pt had no best_wer/patience. Prefer sibling best.pt
            # so a worse later epoch cannot clobber the real best on the next save.
            best_wer, bad = _patience_from_history(history)
            if best_wer >= 1e8 and "dev" in blob:
                best_wer = float(blob["dev"]["wer"])
            best_path = exp_dir / "best.pt"
            if best_path.exists():
                try:
                    best_blob = torch.load(best_path, map_location="cpu", weights_only=False)
                    disk_best = float(
                        best_blob.get("best_wer")
                        or (best_blob.get("dev") or {}).get("wer")
                        or 1e9
                    )
                    if disk_best < best_wer:
                        best_wer = disk_best
                        best_epoch = int(best_blob.get("epoch") or done_epoch)
                        bad = max(0, done_epoch - best_epoch)
                        log.info(
                            "resume: best WER %.2f taken from %s (epoch %d), bad=%d",
                            best_wer * 100,
                            best_path,
                            best_epoch,
                            bad,
                        )
                except Exception as exc:
                    log.warning("resume: could not read %s (%s)", best_path, exc)
        sampler = getattr(train_loader, "batch_sampler", None)
        if sampler is not None and hasattr(sampler, "seed"):
            sampler.seed = int(blob.get("sampler_seed", done_epoch))
        _restore_rng(blob.get("rng"))
        lr_now = cosine_with_warmup(
            global_step,
            warmup=warmup,
            total=total_steps,
            base_lr=float(tcfg["lr"]),
            min_lr=float(tcfg.get("min_lr", 1e-6)),
        )
        for g in opt.param_groups:
            g["lr"] = lr_now
        log.info(
            "resume %s: finished epoch %d, next epoch %d/%d, step=%d lr=%.2e best_wer=%.2f bad=%d",
            resume_path,
            done_epoch,
            start_epoch,
            epochs,
            global_step,
            lr_now,
            best_wer * 100,
            bad,
        )
        if start_epoch > epochs:
            log.info("resume: already completed %d/%d epochs", done_epoch, epochs)
            _save_history(hist_path, history, union_disk=True)
            return exp_dir / "best.pt" if (exp_dir / "best.pt").exists() else resume_path

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        seen = 0
        win_loss, win_n, win_skipped = 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, 1):
            feats = batch["feats"].to(device, non_blocking=True)
            lens = batch["feat_lens"].to(device, non_blocking=True)
            if specaug is not None and epoch >= specaug_start:
                feats = specaug(feats, lens)
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                loss, logits, logit_lens = model.loss(
                    feats,
                    lens,
                    batch["targets"].to(device, non_blocking=True),
                    batch["target_lens"].to(device, non_blocking=True),
                    inter_weight=float(tcfg.get("inter_ctc_weight", 0.5)),
                )
                loss = loss / accum
            scaler.scale(loss).backward()
            if step % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                lr = cosine_with_warmup(
                    global_step,
                    warmup=warmup,
                    total=total_steps,
                    base_lr=float(tcfg["lr"]),
                    min_lr=float(tcfg.get("min_lr", 1e-6)),
                )
                for g in opt.param_groups:
                    g["lr"] = lr
                scale_before = scaler.get_scale() if scaler.is_enabled() else 1.0
                scaler.step(opt)
                scaler.update()
                if scaler.is_enabled() and scaler.get_scale() < scale_before:
                    win_skipped += 1
                opt.zero_grad(set_to_none=True)
                global_step += 1
            batch_loss = float(loss.detach()) * accum * feats.size(0)
            running += batch_loss
            seen += feats.size(0)
            win_loss += batch_loss
            win_n += feats.size(0)
            if step % 50 == 0:
                mem = ""
                if device.type == "cuda":
                    gb = 1024**3
                    mem = "  mem alloc %.1f peak %.1f reserved %.1f GB" % (
                        torch.cuda.memory_allocated(device) / gb,
                        torch.cuda.max_memory_allocated(device) / gb,
                        torch.cuda.memory_reserved(device) / gb,
                    )
                    torch.cuda.reset_peak_memory_stats(device)
                amp = f"  scale {scaler.get_scale():.3g} skipped {win_skipped}" if scaler.is_enabled() else ""
                log.info(
                    "epoch %d step %d loss %.4f (last 50: %.4f) lr %.2e%s%s",
                    epoch,
                    step,
                    running / max(1, seen),
                    win_loss / max(1, win_n),
                    opt.param_groups[0]["lr"],
                    amp,
                    mem,
                )
                win_loss, win_n, win_skipped = 0.0, 0, 0
        dev = evaluate_model(model, dev_loader, tokenizer, device)
        rec = {
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            "dev_wer": dev["wer"],
            "dev_cer": dev["cer"],
            "dev_loss": dev["loss"],
            "seconds": round(time.time() - t0, 1),
            "lr": opt.param_groups[0]["lr"],
            "global_step": global_step,
        }
        history = [h for h in history if int(h.get("epoch", 0)) < epoch]
        history.append(rec)
        history = _save_history(hist_path, history, union_disk=union_hist)
        log.info(
            "epoch %d  train_loss=%.4f  dev_wer=%.2f  dev_cer=%.2f  (%.0fs)",
            epoch,
            rec["train_loss"],
            rec["dev_wer"] * 100,
            rec["dev_cer"] * 100,
            rec["seconds"],
        )
        for i, s in enumerate(dev.get("samples") or []):
            log.info("  sample %d ref: %s", i, (s["ref"] or "")[:160])
            log.info("  sample %d hyp: %s", i, (s["hyp"] or "")[:160])
        if dev["wer"] < best_wer:
            best_wer = dev["wer"]
            bad = 0
            is_best = True
            log.info("  new best WER %.2f", best_wer * 100)
        else:
            bad += 1
            is_best = False
        sampler_seed = getattr(getattr(train_loader, "batch_sampler", None), "seed", 0)
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "config": cfg,
            "vocab_size": tokenizer.vocab_size,
            "dev": dev,
            "global_step": global_step,
            "best_wer": best_wer,
            "bad": bad,
            "total_steps": total_steps,
            "sampler_seed": sampler_seed,
            "history": history,
            "rng": _capture_rng(),
        }
        torch.save(ckpt, exp_dir / "last.pt")
        if is_best:
            torch.save(ckpt, exp_dir / "best.pt")
        if save_epoch_ckpts:
            epoch_dir = exp_dir / "epochs"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, epoch_dir / f"epoch_{epoch:03d}.pt")
        if bad >= patience:
            log.info("early stopping at epoch %d", epoch)
            break

    _save_history(hist_path, history, union_disk=union_hist)
    return exp_dir / "best.pt"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--overfit-batch", action="store_true")
    p.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Load model weights only and start a new run (new optimizer, epoch 1).",
    )
    p.add_argument(
        "--resume",
        nargs="?",
        const="",
        default=None,
        help="Continue from a checkpoint (optimizer, LR schedule, epoch). "
        "If no path is given, uses <exp_dir>/last.pt.",
    )
    args = p.parse_args()
    cfg = load_yaml(args.config)
    resume = args.resume
    if resume is not None and resume == "":
        resume = str(ROOT / cfg["paths"]["exp_dir"] / "last.pt")
    train(cfg, overfit_batch=args.overfit_batch, init_from=args.init_from, resume=resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
