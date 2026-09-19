#!/usr/bin/env python3
"""Evaluate a checkpoint on Dataset.ET or an external jsonl corpus.

    python evaluate.py --config configs/20m_char.yaml --ckpt exp/20m_char/best.pt --split test
    python evaluate.py --config configs/20m_char_12layer.yaml --ckpt exp/20m_char_12layer/best.pt \\
        --jsonl metadata/fleurs_am_et/test.jsonl --decoder greedy
    python evaluate.py --config configs/20m_char_12layer.yaml --onnx exp/20m_char_12layer/model.int8.onnx \\
        --jsonl metadata/fleurs_am_et/test.jsonl --decoder greedy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.dataset import AsrDataset, collate
from data.oov import vocab_from_rows
from data.prepare import load_split_jsonl
from decoding.beam import beam_decode_batch
from decoding.greedy import greedy_decode
from decoding.kenlm import (
    build_decoder,
    decode_batch as kenlm_decode_batch,
    labels_from_tokenizer,
    lm_files,
    load_unigrams,
    resolve_lm_dir,
)
from evaluation.metrics import file_size_mb, score_split
from models.ctc import build_model, param_count
from train import load_cmvn, load_tokenizer
from utils import ROOT, load_jsonl, load_yaml, save_json, setup_logging, write_jsonl

log = setup_logging()


def load_model(cfg: dict, ckpt_path: Path, tokenizer, device: torch.device):
    model = build_model(cfg, tokenizer.vocab_size)
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(blob["model"])
    model.to(device).eval()
    return model


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", default=None, help="PyTorch checkpoint (best.pt). Omit when --onnx is set.")
    p.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="ONNX graph from deployment.export_onnx (fp32 or INT8). Runs on ONNX Runtime.",
    )
    p.add_argument(
        "--onnx-provider",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="ONNX Runtime EP: auto prefers CUDA when the GPU wheel is installed.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Eval batch size (default 8; use 1–2 for ONNX CUDA on 4 GB GPUs).",
    )
    p.add_argument("--split", default="test", choices=["train", "dev", "test"])
    p.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Override metadata jsonl (e.g. metadata/fleurs_am_et/test.jsonl). "
        "clip audio paths are relative to the repo root.",
    )
    p.add_argument(
        "--train-jsonl",
        type=Path,
        default=None,
        help="Train split used for seen/unseen-word WER (default: Dataset.ET train).",
    )
    p.add_argument("--decoder", default="greedy", choices=["greedy", "beam", "kenlm"])
    p.add_argument("--beam-size", type=int, default=8)
    p.add_argument(
        "--token-top-k",
        type=int,
        default=8,
        help="Per-frame logit candidates for prefix beam. 8 is enough without an LM; 50 is the textbook cap.",
    )
    p.add_argument(
        "--lm",
        type=Path,
        default=None,
        help="KenLM .bin/.arpa (default: <exp_dir>/lm_word then <exp_dir>/lm)",
    )
    p.add_argument(
        "--lm-unigrams",
        type=Path,
        default=None,
        help="One word per line (default: same folder as --lm)",
    )
    p.add_argument("--lm-alpha", type=float, default=0.5, help="LM weight for pyctcdecode")
    p.add_argument("--lm-beta", type=float, default=1.5, help="Word-insertion bonus for pyctcdecode")
    p.add_argument("--beam-width", type=int, default=100, help="pyctcdecode beam width (kenlm decoder)")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--match-et",
        default=None,
        choices=["gain", "gain_opus"],
        help="Match FLEURS (or any OOD) audio to Dataset.ET phone/Telegram acoustics: "
        "gain ≈ -25 dBFS RMS; gain_opus also Opus-voip 24 kb/s roundtrip.",
    )
    args = p.parse_args()
    if not args.ckpt and not args.onnx:
        p.error("provide --ckpt or --onnx")

    cfg = load_yaml(args.config)
    exp_dir = ROOT / cfg["paths"]["exp_dir"]
    jsonl = Path(args.jsonl) if args.jsonl else ROOT / "metadata" / f"{args.split}.jsonl"
    if not jsonl.is_absolute():
        jsonl = ROOT / jsonl
    if args.out_dir:
        out_dir = args.out_dir
    else:
        tag = f"{jsonl.parent.name}_{jsonl.stem}" if args.jsonl else args.split
        if args.match_et:
            tag += "_" + args.match_et
        backend = ""
        if args.onnx:
            backend = "_" + Path(args.onnx).stem.replace(".", "_")
        out_dir = exp_dir / f"eval_{tag}_{args.decoder}{backend}"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(cfg, exp_dir)
    cmvn = load_cmvn(cfg, exp_dir, compute_if_missing=False)
    if args.onnx:
        from deployment.onnx_infer import OnnxCTC

        onnx_path = Path(args.onnx)
        if not onnx_path.is_absolute():
            onnx_path = ROOT / onnx_path
        model = OnnxCTC(onnx_path, prefer=args.onnx_provider)
        device = torch.device("cpu")
        log.info("onnx %s  requested=%s  providers=%s", onnx_path, args.onnx_provider, model.providers)
        n_params = None
        model_size = file_size_mb(onnx_path)
        runtime_device = ",".join(model.providers)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(cfg, Path(args.ckpt), tokenizer, device)
        n_params = param_count(model)
        model_size = file_size_mb(args.ckpt)
        runtime_device = str(device)

    rows = load_jsonl(jsonl)
    ds = AsrDataset(
        jsonl,
        tokenizer,
        train=False,
        cmvn=cmvn,
        match_et=args.match_et,
    )
    bs = args.batch_size
    if bs is None:
        using_cuda = args.onnx and "CUDAExecutionProvider" in (
            model.providers if args.onnx else []
        )
        bs = 2 if using_cuda else 8
    loader = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate, num_workers=2)
    if args.onnx:
        log.info("onnx batch_size=%d", bs)

    hyps: list[str] = []
    clip_ids: list[str] = []
    refs_raw: list[str] = []

    if args.decoder == "beam":
        log.info(
            "decoder=beam  beam_size=%d  token_top_k=%d  (CPU prefix search; expect ~1–3 min on dev, not hours)",
            args.beam_size,
            args.token_top_k,
        )

    kenlm_decoder = None
    resolved_lm = None
    if args.decoder == "kenlm":
        if args.lm:
            # The unigram list must come from the same corpus as the LM: pyctcdecode
            # penalises every word missing from it as unknown.
            default_lm, default_uni = Path(args.lm), lm_files(Path(args.lm).parent)[1]
        else:
            default_lm, default_uni = resolve_lm_dir(exp_dir)
        lm_path = args.lm or default_lm
        uni_path = args.lm_unigrams or default_uni
        resolved_lm = lm_path
        if lm_path is None and uni_path is None:
            raise FileNotFoundError(
                f"no LM at {exp_dir / 'lm_word'} or {exp_dir / 'lm'}. Train one first:\n"
                f"  python -m decoding.train_lm --out-dir {exp_dir / 'lm'}"
            )
        unigrams = load_unigrams(uni_path)
        log.info("kenlm unigrams: %s (%d words)", uni_path, len(unigrams or []))
        kenlm_decoder = build_decoder(
            labels_from_tokenizer(tokenizer),
            str(lm_path) if lm_path else None,
            unigrams=unigrams,
            alpha=args.lm_alpha,
            beta=args.lm_beta,
        )
        log.info(
            "decoder=kenlm  lm=%s  unigrams=%s  alpha=%.2f  beta=%.2f  beam_width=%d",
            lm_path,
            uni_path,
            args.lm_alpha,
            args.lm_beta,
            args.beam_width,
        )

    def run_batch(batch):
        feats = batch["feats"].to(device)
        lens = batch["feat_lens"].to(device)
        with torch.inference_mode():
            logits, logit_lens = model(feats, lens)
        if args.decoder == "kenlm":
            return kenlm_decode_batch(kenlm_decoder, logits, logit_lens, beam_width=args.beam_width)
        if args.decoder == "beam":
            return beam_decode_batch(
                logits,
                logit_lens,
                tokenizer,
                beam_size=args.beam_size,
                token_top_k=args.token_top_k,
            )
        return greedy_decode(logits, tokenizer, logit_lens)

    t0_audio = sum(r["duration_s"] for r in rows)
    import time

    t0 = time.perf_counter()
    n_done = 0
    n_total = len(ds)
    for batch in loader:
        hyps.extend(run_batch(batch))
        clip_ids.extend(batch["clip_id"])
        refs_raw.extend(batch["text_raw"])
        n_done += len(batch["clip_id"])
        if args.decoder in ("beam", "kenlm") and (n_done % 64 == 0 or n_done == n_total):
            elapsed = time.perf_counter() - t0
            rate = n_done / max(elapsed, 1e-6)
            eta = (n_total - n_done) / max(rate, 1e-6)
            log.info("%s %d/%d  %.1f utt/s  eta %.0fs", args.decoder, n_done, n_total, rate, eta)
    wall = time.perf_counter() - t0
    for i, (ref, hyp) in enumerate(zip(refs_raw[:4], hyps[:4])):
        log.info("  sample %d ref: %s", i, (ref or "")[:160])
        log.info("  sample %d hyp: %s", i, (hyp or "")[:160])

    train_jsonl = args.train_jsonl
    v_train = None
    if train_jsonl is None:
        default_train = ROOT / "metadata" / "train.jsonl"
        if default_train.exists():
            v_train = set(vocab_from_rows(load_split_jsonl("train")))
        else:
            log.warning("no Dataset.ET metadata/train.jsonl; skipping seen/unseen-word WER")
    else:
        if not train_jsonl.is_absolute():
            train_jsonl = ROOT / train_jsonl
        v_train = set(vocab_from_rows(load_jsonl(train_jsonl)))
    # Align hyps to metadata order via clip_id
    hyp_map = dict(zip(clip_ids, hyps))
    ordered_hyps = [hyp_map.get(r["clip_id"], "") for r in rows]
    scores = score_split(rows, ordered_hyps, v_train=v_train)
    runtime = {
        "rtf": wall / t0_audio if t0_audio else float("nan"),
        "wall_seconds": wall,
        "audio_seconds": t0_audio,
        "parameter_count": n_params,
        "model_size_mb": model_size,
        "decoder": args.decoder,
        "device": runtime_device,
        "onnx": str(args.onnx) if args.onnx else None,
        "lm": str(resolved_lm) if args.decoder == "kenlm" else None,
        "lm_alpha": args.lm_alpha if args.decoder == "kenlm" else None,
        "lm_beta": args.lm_beta if args.decoder == "kenlm" else None,
        "match_et": args.match_et,
    }
    result = {
        "split": jsonl.stem if args.jsonl else args.split,
        "jsonl": str(jsonl),
        "corpus": jsonl.parent.name if args.jsonl else "dataset_et",
        "scores": scores,
        "runtime": runtime,
    }
    save_json(out_dir / "metrics.json", result)
    write_jsonl(
        out_dir / "hypotheses.jsonl",
        [{"clip_id": r["clip_id"], "text": hyp_map.get(r["clip_id"], "")} for r in rows],
    )
    log.info(
        "%s %s  WER=%.2f CER=%.2f unseen_WER=%s  RTF=%.3f",
        jsonl.stem if args.jsonl else args.split,
        args.decoder,
        scores["overall"]["wer_pct"],
        scores["overall"]["cer_pct"],
        scores.get("seen_unseen", {}).get("unseen_word_wer"),
        runtime["rtf"],
    )
    print(json_dumps(result))
    return 0


def json_dumps(obj) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    raise SystemExit(main())
