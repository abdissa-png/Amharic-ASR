#!/usr/bin/env python3
"""Score a Zipformer checkpoint on one or more jsonl test sets.

One encoder pass per batch. Decoders:

* ``ctc``: greedy CTC
* ``rnnt``: greedy transducer
* ``kenlm``: CTC + the word 5-gram LM (pyctcdecode)
* ``ctc_charlm`` / ``ctc_morphlm``: CTC prefix beam + a character or
  morpheme n-gram (``decoding.unit_lm``). Tune α/β on a **dev** jsonl.

The streaming setting matches the demo: chunk 32 and 128 frames of left
context at the 50 Hz embed (640 ms). ``--chunk-size -1`` is the full-context
decode the causal model was also trained on.

KenLM reads the CTC linear layer, not the LogSoftmax wrapper. pyctcdecode
applies log-softmax itself. Alpha 0.5 and beta 0.8 are the weights used for
the published FLEURS numbers. Do not retune word-LM or unit-LM weights on a
test set.

Stop training before running this. The T4 does not fit a training step and
this eval at the same time.

    python eval_zipformer.py \\
        --config configs/zipformer_s_streaming_pool.yaml \\
        --ckpt exp/zipformer_s_streaming_pool/best.pt

    python eval_zipformer.py --no-rnnt --no-kenlm \\
        --char-lm exp/zipformer_s_streaming_pool/lm_char \\
        --morph-lm exp/zipformer_s_streaming_pool/lm_morph \\
        --tune-jsonl metadata/fleurs_am_et/dev.jsonl \\
        --jsonl metadata/fleurs_am_et/test.jsonl
"""

from __future__ import annotations

import argparse
import inspect
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.dataset import AsrDataset, collate
from data.oov import vocab_from_rows
from data.prepare import load_split_jsonl
from decoding.greedy import greedy_decode
from decoding.kenlm import (
    build_decoder,
    decode_batch as kenlm_decode_batch,
    labels_from_tokenizer,
    lm_files,
    load_unigrams,
)
from decoding.unit_lm import decode_batch as unit_lm_decode_batch
from decoding.unit_lm import load_unit_model
from decoding.zipformer_greedy import greedy_search_batch
from evaluation.metrics import score_split
from models.zipformer_icefall import build_zipformer, load_icefall
from train import load_cmvn, load_tokenizer
from utils import ROOT, load_jsonl, load_yaml, save_json, setup_logging, write_jsonl

log = setup_logging()

DEFAULT_JSONL = (
    "metadata/test.jsonl",
    "metadata/fleurs_am_et/test.jsonl",
)
LM_CANDIDATES = (
    "exp/20m_char_12layer/lm_word",
    "exp/lm_word",
)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _find_lm(explicit: str | None) -> tuple[Path, Path | None]:
    if explicit:
        path = _resolve(explicit)
        if path.is_dir():
            model, uni = lm_files(path)
        else:
            model, uni = path, lm_files(path.parent)[1]
        if model is None or not model.exists():
            raise FileNotFoundError(f"no KenLM model at {path}")
        return model, uni
    for folder in LM_CANDIDATES:
        model, uni = lm_files(ROOT / folder)
        if model is not None:
            return model, uni
    raise FileNotFoundError(
        "no KenLM. Expected one of: " + ", ".join(LM_CANDIDATES)
    )


def _pin_streaming(model, chunk: int, left: int) -> None:
    enc = model.encoder
    enc.chunk_size = (chunk,)
    enc.left_context_frames = (left,)


def _write(out_dir: Path, rows: list[dict], hyps: list[str], result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "metrics.json", result)
    write_jsonl(
        out_dir / "hypotheses.jsonl",
        [{"clip_id": r["clip_id"], "text": h} for r, h in zip(rows, hyps)],
    )


TUNE_ALPHAS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
TUNE_BETAS = (0.0, 0.5, 0.8, 1.2)


def encode_jsonl(model, tokenizer, cmvn, args, jsonl: Path, device):
    """One encoder pass. Returns rows, clip order, greedy CTC hyps, RNNT hyps, packed CTC logits."""
    rows = load_jsonl(jsonl)
    ds_kwargs = {"train": False, "cmvn": cmvn, "max_duration": None}
    # Older checkouts of AsrDataset have no eval_speed. 1.0 is a no-op anyway.
    if "eval_speed" in inspect.signature(AsrDataset.__init__).parameters:
        ds_kwargs["eval_speed"] = float(getattr(args, "eval_speed", 1.0))
    ds = AsrDataset(jsonl, tokenizer, **ds_kwargs)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=2
    )
    _pin_streaming(model, args.chunk_size, args.left_context)
    ctc_hyps: list[str] = []
    rnnt_hyps: list[str] = []
    kenlm_hyps: list[str] = []
    clip_ids: list[str] = []
    packed: list[tuple[str, torch.Tensor, int]] = []
    audio = sum(float(r.get("duration_s") or 0.0) for r in rows)
    t0 = time.perf_counter()
    n_done = 0
    for batch in loader:
        feats = batch["feats"].to(device, non_blocking=True)
        lens = batch["feat_lens"].to(device, non_blocking=True)
        with torch.inference_mode():
            encoder_out, encoder_out_lens = model.forward_encoder(feats, lens)
            if args.rnnt:
                for ids in greedy_search_batch(model, encoder_out, encoder_out_lens):
                    rnnt_hyps.append(tokenizer.decode(ids, collapse=False))
            # ctc_output is Dropout, Linear, LogSoftmax. KenLM log-softmaxes
            # again, so it must see the linear layer.
            logits = model.ctc_output[1](encoder_out)
        ctc_hyps.extend(greedy_decode(logits, tokenizer, encoder_out_lens))
        if args.kenlm_decoder is not None:
            kenlm_hyps.extend(
                kenlm_decode_batch(
                    args.kenlm_decoder, logits, encoder_out_lens, beam_width=args.beam_width
                )
            )
        if args.unit_lms:
            cpu = logits.detach().float().cpu()
            for i, cid in enumerate(batch["clip_id"]):
                t = int(encoder_out_lens[i])
                packed.append((cid, cpu[i, :t].contiguous(), t))
        clip_ids.extend(batch["clip_id"])
        n_done += len(batch["clip_id"])
        if n_done % 64 == 0 or n_done == len(ds):
            elapsed = time.perf_counter() - t0
            rate = n_done / max(elapsed, 1e-6)
            log.info(
                "%s %d/%d  %.1f utt/s  eta %.0fs",
                jsonl.parent.name,
                n_done,
                len(ds),
                rate,
                (len(ds) - n_done) / max(rate, 1e-6),
            )
    wall = time.perf_counter() - t0
    log.info(
        "%s encoded  %d utt  %.0fs audio  %.0fs wall  chunk=%d left=%d",
        jsonl.parent.name,
        len(rows),
        audio,
        wall,
        args.chunk_size,
        args.left_context,
    )
    return {
        "rows": rows,
        "clip_ids": clip_ids,
        "ctc_hyps": ctc_hyps,
        "rnnt_hyps": rnnt_hyps,
        "kenlm_hyps": kenlm_hyps,
        "packed": packed,
        "audio": audio,
        "wall": wall,
    }


def _ordered(hyps: list[str], clip_ids: list[str], rows: list[dict]) -> list[str]:
    order = {cid: i for i, cid in enumerate(clip_ids)}
    if len(order) != len(rows):
        raise RuntimeError(f"decoded {len(order)} clips, jsonl has {len(rows)}")
    return [hyps[order[r["clip_id"]]] for r in rows]


def _decode_unit_packed(packed, tokenizer, km, model, unit: str, *, alpha: float, beta: float, args) -> dict[str, str]:
    hyps: dict[str, str] = {}
    t0 = time.perf_counter()
    for i, (cid, logit, _t) in enumerate(packed, 1):
        hyps[cid] = unit_lm_decode_batch(
            logit.unsqueeze(0),
            torch.tensor([logit.size(0)]),
            tokenizer,
            km,
            model,
            unit=unit,
            beam_size=args.unit_beam,
            token_top_k=args.unit_top_k,
            alpha=alpha,
            beta=beta,
        )[0]
        if i % 32 == 0 or i == len(packed):
            rate = i / max(time.perf_counter() - t0, 1e-6)
            log.info("  %s %d/%d  %.2f utt/s", unit, i, len(packed), rate)
    return hyps


def tune_unit_lm(name: str, packed, rows, tokenizer, km, model, unit: str, v_train, args) -> tuple[float, float, float]:
    """Pick α then β on this split's folded WER. Never call this on a test set."""
    saved_beam, saved_k = args.unit_beam, args.unit_top_k
    args.unit_beam = min(args.unit_beam, 16)
    args.unit_top_k = min(args.unit_top_k, 8)
    try:
        beta = args.unit_beta
        best_a, best_wer = args.unit_alpha, float("inf")
        log.info("tune %s alpha  beta=%.2f  n=%d  beam=%d", name, beta, len(rows), args.unit_beam)
        for alpha in TUNE_ALPHAS:
            hyps = _decode_unit_packed(packed, tokenizer, km, model, unit, alpha=alpha, beta=beta, args=args)
            ordered = [hyps[r["clip_id"]] for r in rows]
            wer = score_split(rows, ordered, v_train=None, fold_geez=True)["overall"]["wer"]
            log.info("  alpha=%.2f beta=%.2f  folded WER %.2f", alpha, beta, 100 * wer)
            if wer < best_wer:
                best_a, best_wer = alpha, wer
        best_b = beta
        log.info("tune %s beta  alpha=%.2f", name, best_a)
        for beta in TUNE_BETAS:
            hyps = _decode_unit_packed(packed, tokenizer, km, model, unit, alpha=best_a, beta=beta, args=args)
            ordered = [hyps[r["clip_id"]] for r in rows]
            wer = score_split(rows, ordered, v_train=None, fold_geez=True)["overall"]["wer"]
            log.info("  alpha=%.2f beta=%.2f  folded WER %.2f", best_a, beta, 100 * wer)
            if wer < best_wer:
                best_b, best_wer = beta, wer
        log.info("tune %s chose alpha=%.2f beta=%.2f  dev WER %.2f", name, best_a, best_b, 100 * best_wer)
        return best_a, best_b, best_wer
    finally:
        args.unit_beam, args.unit_top_k = saved_beam, saved_k


def eval_jsonl(model, tokenizer, cmvn, args, jsonl: Path, device, epoch, v_train) -> None:
    blob = encode_jsonl(model, tokenizer, cmvn, args, jsonl, device)
    rows = blob["rows"]
    wall = blob["wall"]
    audio = blob["audio"]

    tag = f"{jsonl.parent.name}_{jsonl.stem}_chunk{args.chunk_size}"
    exp_dir = _resolve(args.exp_dir) if args.exp_dir else _resolve(load_yaml(args.config)["paths"]["exp_dir"])
    systems = [("ctc", _ordered(blob["ctc_hyps"], blob["clip_ids"], rows), None, None, None)]
    if args.rnnt:
        systems.append(("rnnt", _ordered(blob["rnnt_hyps"], blob["clip_ids"], rows), None, None, None))
    if args.kenlm_decoder is not None:
        systems.append(
            (
                "ctc_kenlm",
                _ordered(blob["kenlm_hyps"], blob["clip_ids"], rows),
                str(args.lm_path),
                args.lm_alpha,
                args.lm_beta,
            )
        )
    for spec in args.unit_lms:
        t0 = time.perf_counter()
        hyps_map = _decode_unit_packed(
            blob["packed"],
            tokenizer,
            spec["km"],
            spec["model"],
            spec["unit"],
            alpha=spec["alpha"],
            beta=spec["beta"],
            args=args,
        )
        ordered = [hyps_map[r["clip_id"]] for r in rows]
        systems.append((spec["name"], ordered, str(spec["path"]), spec["alpha"], spec["beta"]))
        log.info("%s decode wall %.0fs", spec["name"], time.perf_counter() - t0)

    log.info("%s  %d utt  %.0fs audio  encode %.0fs  chunk=%d left=%d", tag, len(rows), audio, wall, args.chunk_size, args.left_context)
    for name, hyps, lm, alpha, beta in systems:
        folded = score_split(rows, hyps, v_train=v_train, fold_geez=True)
        unfolded = score_split(rows, hyps, v_train=None, fold_geez=False)
        result = {
            "split": jsonl.stem,
            "jsonl": str(jsonl),
            "corpus": jsonl.parent.name,
            "ckpt": str(_resolve(args.ckpt)),
            "epoch": epoch,
            "chunk_size": args.chunk_size,
            "left_context_frames": args.left_context,
            "scores_folded": folded,
            "scores_unfolded": unfolded,
            "runtime": {
                "wall_seconds": wall,
                "audio_seconds": audio,
                "n": len(rows),
                "lm": lm,
                "lm_alpha": alpha if lm else None,
                "lm_beta": beta if lm else None,
                "beam_width": args.beam_width if name == "ctc_kenlm" else (args.unit_beam if lm else None),
            },
        }
        _write(exp_dir / f"eval_{tag}_{name}", rows, hyps, result)
        f = folded["overall"]
        u = unfolded["overall"]
        log.info(
            "%s folded WER %.2f CER %.2f | unfolded WER %.2f CER %.2f",
            name,
            f["wer_pct"],
            f["cer_pct"],
            u["wer_pct"],
            u["cer_pct"],
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/zipformer_s_streaming_pool.yaml")
    p.add_argument("--ckpt", default="exp/zipformer_s_streaming_pool/best.pt")
    p.add_argument("--exp-dir", default=None, help="where metrics are written; default is the config exp_dir")
    p.add_argument("--jsonl", action="append", default=None, help="repeat for each test set")
    p.add_argument("--lm", default=None, help="lm.bin or a folder containing it")
    p.add_argument("--lm-alpha", type=float, default=0.5)
    p.add_argument("--lm-beta", type=float, default=0.8)
    p.add_argument("--beam-width", type=int, default=100)
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--left-context", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--eval-speed",
        type=float,
        default=1.0,
        help="Time-stretch factor at eval (>1 is faster speech). Training only saw 0.9–1.1.",
    )
    p.add_argument("--no-rnnt", action="store_true")
    p.add_argument("--no-kenlm", action="store_true")
    p.add_argument("--char-lm", default=None, help="folder with a character KenLM (lm.bin / lm.arpa)")
    p.add_argument("--morph-lm", default=None, help="folder with a morpheme KenLM")
    p.add_argument("--unit-alpha", type=float, default=0.6)
    p.add_argument("--unit-beta", type=float, default=0.8)
    p.add_argument("--unit-beam", type=int, default=32)
    p.add_argument("--unit-top-k", type=int, default=12)
    p.add_argument(
        "--tune-jsonl",
        default=None,
        help="dev jsonl used to pick unit-LM α/β; never a test set",
    )
    args = p.parse_args()
    args.rnnt = not args.no_rnnt

    cfg = load_yaml(_resolve(args.config))
    exp_dir = _resolve(args.exp_dir) if args.exp_dir else _resolve(cfg["paths"]["exp_dir"])
    ckpt_path = _resolve(args.ckpt)
    jsonls = [_resolve(j) for j in (args.jsonl or DEFAULT_JSONL)]
    missing = [str(j) for j in jsonls if not j.exists()]
    if missing:
        raise FileNotFoundError("missing jsonl: " + ", ".join(missing))

    load_icefall()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(cfg, exp_dir)
    cmvn = load_cmvn(cfg, exp_dir, compute_if_missing=False)
    model = build_zipformer(cfg, tokenizer.vocab_size)
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    model.to(device).eval()
    epoch = blob.get("epoch")
    log.info("ckpt %s epoch=%s device=%s vocab=%d", ckpt_path, epoch, device, tokenizer.vocab_size)

    args.kenlm_decoder = None
    args.lm_path = None
    if not args.no_kenlm:
        lm_path, uni_path = _find_lm(args.lm)
        args.lm_path = lm_path
        unigrams = load_unigrams(uni_path)
        log.info("kenlm %s unigrams=%s (%d) alpha=%.2f beta=%.2f beam=%d", lm_path, uni_path, len(unigrams or []), args.lm_alpha, args.lm_beta, args.beam_width)
        args.kenlm_decoder = build_decoder(
            labels_from_tokenizer(tokenizer),
            str(lm_path),
            unigrams=unigrams,
            alpha=args.lm_alpha,
            beta=args.lm_beta,
        )

    args.unit_lms = []
    if args.char_lm:
        km, umodel, path = load_unit_model(_resolve(args.char_lm))
        args.unit_lms.append(
            {
                "name": "ctc_charlm",
                "km": km,
                "model": umodel,
                "path": path,
                "unit": "char",
                "alpha": args.unit_alpha,
                "beta": args.unit_beta,
            }
        )
        log.info("char-lm %s order=%s", path, umodel.order() if callable(umodel.order) else umodel.order)
    if args.morph_lm:
        km, umodel, path = load_unit_model(_resolve(args.morph_lm))
        args.unit_lms.append(
            {
                "name": "ctc_morphlm",
                "km": km,
                "model": umodel,
                "path": path,
                "unit": "morph",
                "alpha": args.unit_alpha,
                "beta": args.unit_beta,
            }
        )
        log.info("morph-lm %s order=%s", path, umodel.order() if callable(umodel.order) else umodel.order)

    v_train = None
    train_jsonl = ROOT / "metadata" / "train.jsonl"
    if train_jsonl.exists():
        v_train = set(vocab_from_rows(load_split_jsonl("train")))

    if args.tune_jsonl:
        if not args.unit_lms:
            raise SystemExit("--tune-jsonl needs --char-lm and/or --morph-lm")
        tune_path = _resolve(args.tune_jsonl)
        if not tune_path.exists():
            raise FileNotFoundError(tune_path)
        if tune_path.stem == "test" or tune_path.resolve() in {j.resolve() for j in jsonls}:
            raise SystemExit("do not tune unit-LM weights on a test jsonl")
        log.info("tuning unit LMs on %s (greedy CTC still uses the test sets below)", tune_path)
        encoded = encode_jsonl(model, tokenizer, cmvn, args, tune_path, device)
        for spec in args.unit_lms:
            a, b, wer = tune_unit_lm(
                spec["name"],
                encoded["packed"],
                encoded["rows"],
                tokenizer,
                spec["km"],
                spec["model"],
                spec["unit"],
                v_train,
                args,
            )
            spec["alpha"], spec["beta"], spec["dev_wer"] = a, b, wer
            save_json(
                exp_dir / f"tune_{tune_path.parent.name}_{tune_path.stem}_{spec['name']}.json",
                {
                    "jsonl": str(tune_path),
                    "lm": str(spec["path"]),
                    "unit": spec["unit"],
                    "alpha": a,
                    "beta": b,
                    "dev_wer_folded": wer,
                    "beam": args.unit_beam,
                    "token_top_k": args.unit_top_k,
                },
            )

    for jsonl in jsonls:
        eval_jsonl(model, tokenizer, cmvn, args, jsonl, device, epoch, v_train)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
