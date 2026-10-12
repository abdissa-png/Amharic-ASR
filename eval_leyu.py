#!/usr/bin/env python3
"""Score Leyu with CTC greedy and the word 5-gram.

No character LM. Alpha 0.5 and beta 0.8 are the FLEURS word-LM weights.
This script does not retune them. Dev is the default split; the test split
is scored only with --test.

Leyu clips run to about 80 s, so the default batch is 2. The report is
overall and by dialect (the ``region`` field written by prepare_leyu).

    python eval_leyu.py --lm exp/20m_char_12layer/lm_word   
    python eval_leyu.py --test --lm /path/to/lm_word
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from decoding.kenlm import build_decoder, labels_from_tokenizer, load_unigrams
from eval_zipformer import _find_lm, _ordered, _resolve, encode_jsonl
from evaluation.wer import score_texts
from models.zipformer_icefall import build_zipformer, load_icefall
from train import load_cmvn, load_tokenizer
from utils import ROOT, load_yaml, save_json, setup_logging, write_jsonl

log = setup_logging()


def _dialect(row: dict) -> str:
    region = str(row.get("region") or "").strip()
    if region:
        return region
    parts = str(row.get("clip_id") or "").split("_")
    if len(parts) >= 2 and parts[0] == "leyu":
        return parts[1]
    return "unknown"


def _emit(name: str, rows: list[dict], hyps: list[str]) -> dict:
    refs = [r.get("text_raw") or r["text"] for r in rows]
    overall = score_texts(refs, hyps, fold_geez=True)
    by_dialect: dict[str, dict] = {}
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[_dialect(row)].append(i)
    for dialect in sorted(groups):
        idx = groups[dialect]
        by_dialect[dialect] = score_texts(
            [refs[i] for i in idx],
            [hyps[i] for i in idx],
            fold_geez=True,
        )
        by_dialect[dialect]["n_utt"] = len(idx)
    log.info(
        "%s  WER %.2f  CER %.2f  S %d  D %d  I %d  H %d  n %d",
        name,
        overall["wer"] * 100,
        overall["cer"] * 100,
        overall["substitutions"],
        overall["deletions"],
        overall["insertions"],
        overall["hits"],
        overall["n"],
    )
    for dialect, scored in by_dialect.items():
        log.info(
            "  %-12s  WER %.2f  CER %.2f  S %d  D %d  I %d  utt %d",
            dialect,
            scored["wer"] * 100,
            scored["cer"] * 100,
            scored["substitutions"],
            scored["deletions"],
            scored["insertions"],
            scored["n_utt"],
        )
    return {"overall": overall, "by_dialect": by_dialect}


def main() -> int:
    p = argparse.ArgumentParser(description="Leyu dev/test with the word 5-gram LM")
    p.add_argument("--config", default="configs/zipformer_s_streaming_pool.yaml")
    p.add_argument("--ckpt", default="exp/zipformer_s_streaming_pool/best.pt")
    p.add_argument("--exp-dir", default=None)
    p.add_argument("--lm", default=None, help="lm.bin or a folder containing it")
    p.add_argument("--lm-alpha", type=float, default=0.5)
    p.add_argument("--lm-beta", type=float, default=0.8)
    p.add_argument("--beam-width", type=int, default=100)
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--left-context", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--no-rnnt", action="store_true")
    p.add_argument("--test", action="store_true", help="score metadata/leyu_am/test.jsonl too")
    p.add_argument("--jsonl", action="append", default=None, help="override the Leyu split path")
    args = p.parse_args()
    args.rnnt = not args.no_rnnt
    args.eval_speed = 1.0
    args.unit_lms = []

    cfg = load_yaml(_resolve(args.config))
    exp_dir = _resolve(args.exp_dir) if args.exp_dir else _resolve(cfg["paths"]["exp_dir"])
    if args.jsonl:
        jsonls = [_resolve(j) for j in args.jsonl]
    else:
        jsonls = [ROOT / "metadata" / "leyu_am" / "dev.jsonl"]
        if args.test:
            jsonls.append(ROOT / "metadata" / "leyu_am" / "test.jsonl")
    missing = [str(j) for j in jsonls if not j.exists()]
    if missing:
        raise FileNotFoundError(
            "missing Leyu jsonl: "
            + ", ".join(missing)
            + ". Run python -m data.prepare_leyu first."
        )

    load_icefall()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(cfg, exp_dir)
    cmvn = load_cmvn(cfg, exp_dir, compute_if_missing=False)
    model = build_zipformer(cfg, tokenizer.vocab_size)
    ckpt_path = _resolve(args.ckpt)
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    model.to(device).eval()
    epoch = blob.get("epoch")
    log.info("ckpt %s epoch=%s device=%s", ckpt_path, epoch, device)

    lm_path, uni_path = _find_lm(args.lm)
    unigrams = load_unigrams(uni_path)
    log.info(
        "kenlm %s unigrams=%d alpha=%.2f beta=%.2f beam=%d",
        lm_path,
        len(unigrams or []),
        args.lm_alpha,
        args.lm_beta,
        args.beam_width,
    )
    args.kenlm_decoder = build_decoder(
        labels_from_tokenizer(tokenizer),
        str(lm_path),
        unigrams=unigrams,
        alpha=args.lm_alpha,
        beta=args.lm_beta,
    )
    args.lm_path = lm_path

    for jsonl in jsonls:
        blob_enc = encode_jsonl(model, tokenizer, cmvn, args, jsonl, device)
        rows = blob_enc["rows"]
        systems = {
            "ctc": _ordered(blob_enc["ctc_hyps"], blob_enc["clip_ids"], rows),
            "ctc_kenlm": _ordered(blob_enc["kenlm_hyps"], blob_enc["clip_ids"], rows),
        }
        if args.rnnt:
            systems["rnnt"] = _ordered(blob_enc["rnnt_hyps"], blob_enc["clip_ids"], rows)
        tag = f"leyu_am_{jsonl.stem}_chunk{args.chunk_size}"
        summary = {
            "jsonl": str(jsonl),
            "ckpt": str(ckpt_path),
            "epoch": epoch,
            "chunk_size": args.chunk_size,
            "left_context_frames": args.left_context,
            "lm": str(lm_path),
            "lm_alpha": args.lm_alpha,
            "lm_beta": args.lm_beta,
            "beam_width": args.beam_width,
            "systems": {},
        }
        for name, hyps in systems.items():
            log.info("%s %s", tag, name)
            summary["systems"][name] = _emit(name, rows, hyps)
            out = exp_dir / f"eval_{tag}_{name}"
            out.mkdir(parents=True, exist_ok=True)
            write_jsonl(
                out / "hypotheses.jsonl",
                [{"clip_id": r["clip_id"], "text": h, "region": _dialect(r)} for r, h in zip(rows, hyps)],
            )
        save_json(exp_dir / f"eval_{tag}.json", summary)
        log.info("wrote %s", exp_dir / f"eval_{tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
