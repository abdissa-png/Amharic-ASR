#!/usr/bin/env python3
"""Score FLEURS the way the demo microphone stream works.

``eval_zipformer.py`` pins chunk 32 / left 128 and runs one
``forward_encoder``. The live demo does not: it pumps ~250 ms of PCM into
``ZipformerStream``, keeps the icefall encoder cache, and on stop flushes a
1 s room-noise tail. Transducer tokens come from ``greedy_search_continue``
per embed chunk; CTC (greedy / word LM / char LM) is run on the concatenated
encoder output only at the end, same as ``demo.app`` ``/stream?final=1``.

Do not retune α/β here. Char LM uses the FLEURS-dev weights (0.4 / 1.2).

    python eval_zipformer_stream.py \\
        --config configs/zipformer_s_streaming_pool.yaml \\
        --ckpt exp/zipformer_s_streaming_pool/best.pt \\
        --jsonl metadata/fleurs_am_et/test.jsonl
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

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
from decoding.zipformer_stream import ZipformerStream
from evaluation.metrics import score_split
from models.zipformer_icefall import build_zipformer, load_icefall
from train import load_cmvn, load_tokenizer
from utils import ROOT, load_jsonl, load_yaml, save_json, setup_logging, write_jsonl

log = setup_logging()

DEFAULT_JSONL = "metadata/fleurs_am_et/test.jsonl"
LM_CANDIDATES = (
    "exp/20m_char_12layer/lm_word",
    "exp/lm_word",
)
CHAR_LM_DEFAULT = "exp/zipformer_s_streaming_pool/lm_char"
# Demo /stream pump interval and FLEURS-dev char LM weights.
PUMP_MS = 250
CHAR_ALPHA = 0.4
CHAR_BETA = 1.2
CHAR_BEAM = 32
CHAR_TOP_K = 12


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


def _write(out_dir: Path, rows: list[dict], hyps: list[str], result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "metrics.json", result)
    write_jsonl(
        out_dir / "hypotheses.jsonl",
        [{"clip_id": r["clip_id"], "text": h} for r, h in zip(rows, hyps)],
    )


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=-1)
    return np.ascontiguousarray(wav, dtype=np.float32), int(sr)


def stream_utterance(
    model,
    cmvn,
    wav: np.ndarray,
    sample_rate: int,
    device: torch.device,
    pump_ms: int,
) -> ZipformerStream:
    """Feed PCM in demo-sized pumps, then flush. Matches ``/stream``."""
    sess = ZipformerStream(model, cmvn, device)
    n = int(wav.shape[0])
    if pump_ms <= 0:
        sess.add(wav, sample_rate, 0, final=True)
        return sess
    hop = max(1, int(sample_rate * pump_ms / 1000.0))
    offset = 0
    while offset < n:
        end = min(offset + hop, n)
        sess.add(wav[offset:end], sample_rate, offset, final=end >= n)
        offset = end
    if not sess.flushed:
        sess.add(np.zeros(0, dtype=np.float32), sample_rate, offset, final=True)
    return sess


def decode_session(model, tokenizer, sess: ZipformerStream, args) -> dict[str, str]:
    out = {"rnnt": tokenizer.decode(sess.tokens, collapse=False) if sess.tokens else ""}
    if sess.encoder_out is None:
        out["ctc"] = ""
        out["ctc_kenlm"] = ""
        out["ctc_charlm"] = ""
        return out
    encoder_out = sess.encoder_out
    lens = torch.tensor([encoder_out.size(1)], device=encoder_out.device)
    logits = model.ctc_output[1](encoder_out)
    out["ctc"] = greedy_decode(logits, tokenizer, lens)[0]
    if args.kenlm_decoder is not None:
        out["ctc_kenlm"] = kenlm_decode_batch(
            args.kenlm_decoder, logits, lens, beam_width=args.beam_width
        )[0]
    else:
        out["ctc_kenlm"] = ""
    if args.char_km is not None:
        out["ctc_charlm"] = unit_lm_decode_batch(
            logits,
            lens,
            tokenizer,
            args.char_km,
            args.char_model,
            unit="char",
            beam_size=args.unit_beam,
            token_top_k=args.unit_top_k,
            alpha=args.unit_alpha,
            beta=args.unit_beta,
        )[0]
    else:
        out["ctc_charlm"] = ""
    return out


def eval_jsonl(model, tokenizer, cmvn, args, jsonl: Path, device, epoch, v_train) -> None:
    rows = load_jsonl(jsonl)
    if args.limit:
        rows = rows[: args.limit]
    exp_dir = _resolve(args.exp_dir) if args.exp_dir else _resolve(load_yaml(args.config)["paths"]["exp_dir"])
    tag = f"{jsonl.parent.name}_{jsonl.stem}_stream"
    names = ["ctc"]
    if args.rnnt:
        names.append("rnnt")
    if args.kenlm_decoder is not None:
        names.append("ctc_kenlm")
    if args.char_km is not None:
        names.append("ctc_charlm")
    hyps = {name: [] for name in names}
    audio = 0.0
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i, row in enumerate(rows, 1):
            path = ROOT / row["audio"]
            wav, sr = _load_wav(path)
            audio += wav.shape[0] / max(sr, 1)
            sess = stream_utterance(model, cmvn, wav, sr, device, args.pump_ms)
            texts = decode_session(model, tokenizer, sess, args)
            for name in names:
                hyps[name].append(texts[name])
            if i % 32 == 0 or i == len(rows):
                elapsed = time.perf_counter() - t0
                rate = i / max(elapsed, 1e-6)
                log.info(
                    "%s %d/%d  %.2f utt/s  eta %.0fs",
                    jsonl.parent.name,
                    i,
                    len(rows),
                    rate,
                    (len(rows) - i) / max(rate, 1e-6),
                )
    wall = time.perf_counter() - t0
    left = int(model.encoder.left_context_frames[0])
    log.info(
        "%s  %d utt  %.0fs audio  %.0fs wall  pump=%dms left=%d",
        tag,
        len(rows),
        audio,
        wall,
        args.pump_ms,
        left,
    )
    systems = [("ctc", hyps["ctc"], None, None, None)]
    if args.rnnt:
        systems.append(("rnnt", hyps["rnnt"], None, None, None))
    if args.kenlm_decoder is not None:
        systems.append(
            ("ctc_kenlm", hyps["ctc_kenlm"], str(args.lm_path), args.lm_alpha, args.lm_beta)
        )
    if args.char_km is not None:
        systems.append(
            (
                "ctc_charlm",
                hyps["ctc_charlm"],
                str(args.char_lm_path),
                args.unit_alpha,
                args.unit_beta,
            )
        )
    for name, texts, lm, alpha, beta in systems:
        folded = score_split(rows, texts, v_train=v_train, fold_geez=True)
        unfolded = score_split(rows, texts, v_train=None, fold_geez=False)
        result = {
            "split": jsonl.stem,
            "jsonl": str(jsonl),
            "corpus": jsonl.parent.name,
            "ckpt": str(_resolve(args.ckpt)),
            "epoch": epoch,
            "mode": "demo_stream",
            "pump_ms": args.pump_ms,
            "flush_tail_s": 1.0,
            "left_context_frames": left,
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
        _write(exp_dir / f"eval_{tag}_{name}", rows, texts, result)
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
    p.add_argument("--exp-dir", default=None)
    p.add_argument("--jsonl", default=DEFAULT_JSONL)
    p.add_argument("--lm", default=None)
    p.add_argument("--lm-alpha", type=float, default=0.5)
    p.add_argument("--lm-beta", type=float, default=0.8)
    p.add_argument("--beam-width", type=int, default=100)
    p.add_argument("--pump-ms", type=int, default=PUMP_MS)
    p.add_argument("--limit", type=int, default=0, help="decode only the first N clips")
    p.add_argument("--no-rnnt", action="store_true")
    p.add_argument("--no-kenlm", action="store_true")
    p.add_argument("--no-char-lm", action="store_true")
    p.add_argument("--char-lm", default=CHAR_LM_DEFAULT)
    p.add_argument("--unit-alpha", type=float, default=CHAR_ALPHA)
    p.add_argument("--unit-beta", type=float, default=CHAR_BETA)
    p.add_argument("--unit-beam", type=int, default=CHAR_BEAM)
    p.add_argument("--unit-top-k", type=int, default=CHAR_TOP_K)
    args = p.parse_args()
    args.rnnt = not args.no_rnnt

    cfg = load_yaml(_resolve(args.config))
    exp_dir = _resolve(args.exp_dir) if args.exp_dir else _resolve(cfg["paths"]["exp_dir"])
    ckpt_path = _resolve(args.ckpt)
    jsonl = _resolve(args.jsonl)
    if not jsonl.exists():
        raise FileNotFoundError(jsonl)

    load_icefall()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(cfg, exp_dir)
    cmvn = load_cmvn(cfg, exp_dir, compute_if_missing=False)
    model = build_zipformer(cfg, tokenizer.vocab_size)
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    model.to(device).eval()
    epoch = blob.get("epoch")
    log.info(
        "ckpt %s epoch=%s device=%s vocab=%d left=%s",
        ckpt_path,
        epoch,
        device,
        tokenizer.vocab_size,
        model.encoder.left_context_frames,
    )

    args.kenlm_decoder = None
    args.lm_path = None
    if not args.no_kenlm:
        lm_path, uni_path = _find_lm(args.lm)
        args.lm_path = lm_path
        unigrams = load_unigrams(uni_path)
        log.info(
            "kenlm %s unigrams=%s (%d) alpha=%.2f beta=%.2f beam=%d",
            lm_path,
            uni_path,
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

    args.char_km = None
    args.char_model = None
    args.char_lm_path = None
    if not args.no_char_lm:
        km, umodel, path = load_unit_model(_resolve(args.char_lm))
        args.char_km, args.char_model, args.char_lm_path = km, umodel, path
        log.info("char-lm %s alpha=%.2f beta=%.2f", path, args.unit_alpha, args.unit_beta)

    v_train = None
    train_jsonl = ROOT / "metadata" / "train.jsonl"
    if train_jsonl.exists():
        v_train = set(vocab_from_rows(load_split_jsonl("train")))

    eval_jsonl(model, tokenizer, cmvn, args, jsonl, device, epoch, v_train)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
