#!/usr/bin/env python3
"""Why the streaming Zipformer drops or swaps words on a timed clip.

Runs the demo stream encoder, then lines up log-mel energy, CTC blank mass,
entropy, encoder L2, RNNT emissions, and streaming-chunk edges against a
YouTube (or other) SRT reference.

    python -m evaluation.inspect_internals \\
        --wav exp/zipformer_s_streaming_pool/youtube_timed/byvT14SJHHI.wav \\
        --ref-srt exp/zipformer_s_streaming_pool/youtube_timed/yt_subs/byvT14SJHHI.am.srt
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from decoding.greedy import greedy_decode
from decoding.kenlm import build_decoder, labels_from_tokenizer, load_unigrams
from decoding.zipformer_greedy import greedy_search_batch
from eval_youtube_timed import (
    FRAME_S,
    _SRT_CLOCK,
    _SRT_TS,
    _TAG,
    kenlm_words,
)
from eval_zipformer_stream import _find_lm, _load_wav, _resolve, stream_utterance
from evaluation.error_taxonomy import _align
from evaluation.geez_eval import normalize as geez_normalize
from models.features import SAMPLE_RATE, log_mel
from models.zipformer_icefall import build_zipformer, load_icefall
from train import load_cmvn, load_tokenizer
from utils import load_yaml, save_json, setup_logging

log = setup_logging()

# Zipformer output is 40 ms/frame; a streaming embed chunk is 32 embed frames
# = 16 encoder frames = 640 ms, the FEATURE_SHIFT the cache advances by.
CHUNK_FRAMES = 16
CLOCK = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)")


def _sec(stamp: str) -> float:
    h, m, s, ms = (int(x) for x in CLOCK.search(stamp).groups())
    return h * 3600 + m * 60 + s + ms / 1000.0


def youtube_words(path: Path) -> list[dict]:
    """Flatten rolling auto-captions into timed words.

    YouTube writes each phrase twice, then grows it. Process *lines*, not
    whole cues, or the prefix is concatenated twice.
    """
    lines: list[tuple[float, float, str]] = []
    start = end = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or _SRT_TS.match(line):
            continue
        if _SRT_CLOCK.match(line):
            left, right = re.split(r"\s*-->\s*", line)
            start, end = _sec(left), _sec(right)
            continue
        text = geez_normalize(_TAG.sub("", line).replace("&nbsp;", " "), fold_geez=True)
        if text and start is not None:
            lines.append((start, end, text))

    flat: list[tuple[float, float, str]] = []
    for s, e, text in lines:
        if not flat:
            flat.append((s, e, text))
            continue
        ps, pe, pt = flat[-1]
        if text == pt:
            flat[-1] = (ps, e, pt)
        elif text.startswith(pt):
            flat[-1] = (ps, e, text)
        elif pt.startswith(text):
            continue
        else:
            flat.append((s, e, text))

    words: list[dict] = []
    prev_end = 0.0
    for s, e, text in flat:
        toks = text.split()
        if not toks:
            continue
        s = max(s, prev_end)
        if e <= s:
            e = s + 0.08 * len(toks)
        step = (e - s) / len(toks)
        for i, w in enumerate(toks):
            words.append(
                {
                    "text": w,
                    "start_s": round(s + i * step, 3),
                    "end_s": round(s + (i + 1) * step, 3),
                }
            )
        prev_end = words[-1]["end_s"]
    return words


def _window(frames: np.ndarray, t0: float, t1: float, frame_s: float) -> slice:
    a = max(0, int(t0 / frame_s))
    b = min(len(frames), max(a + 1, int(math.ceil(t1 / frame_s))))
    return slice(a, b)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/zipformer_s_streaming_pool.yaml")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--wav", required=True)
    p.add_argument("--ref-srt", required=True)
    p.add_argument("--pump-ms", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = load_yaml(_resolve(args.config))
    exp_dir = _resolve(cfg["paths"]["exp_dir"])
    ckpt_path = _resolve(args.ckpt) if args.ckpt else exp_dir / "best.pt"
    out = _resolve(args.out) if args.out else exp_dir / "internals.json"

    wav, sr = _load_wav(_resolve(args.wav))
    duration = wav.shape[0] / max(sr, 1)
    ref_words = youtube_words(_resolve(args.ref_srt))
    log.info("wav %.1fs  ref words %d", duration, len(ref_words))

    load_icefall()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(cfg, exp_dir)
    cmvn = load_cmvn(cfg, exp_dir, compute_if_missing=False)
    model = build_zipformer(cfg, tokenizer.vocab_size)
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    model.to(device).eval()

    lm_path, uni_path = _find_lm(None)
    kenlm = build_decoder(
        labels_from_tokenizer(tokenizer),
        str(lm_path),
        unigrams=load_unigrams(uni_path),
        alpha=0.5,
        beta=0.8,
    )

    with torch.inference_mode():
        sess = stream_utterance(model, cmvn, wav, sr, device, args.pump_ms)
        encoder_out = sess.encoder_out
        lens = torch.tensor([encoder_out.size(1)], device=device)
        timed = greedy_search_batch(model, encoder_out, lens, return_timestamps=True)
        logits = model.ctc_output[1](encoder_out)
        ctc_text = greedy_decode(logits, tokenizer, lens)[0]
        kenlm_text, hyp_words = kenlm_words(
            kenlm, logits, int(lens[0]), 100, frame_s=FRAME_S
        )
        log_probs = F.log_softmax(logits[0], dim=-1)
        probs = log_probs.exp()
        blank_p = probs[:, tokenizer.blank_id].detach().float().cpu().numpy()
        entropy = (-(probs * log_probs).sum(-1)).detach().float().cpu().numpy()
        top_p, top_i = probs.max(dim=-1)
        top_p = top_p.detach().float().cpu().numpy()
        top_i = top_i.detach().cpu().numpy()
        enc_l2 = encoder_out[0].float().norm(dim=-1).cpu().numpy()

    wav_t = torch.from_numpy(wav)
    if sr != SAMPLE_RATE:
        import torchaudio

        wav_t = torchaudio.functional.resample(wav_t.unsqueeze(0), sr, SAMPLE_RATE).squeeze(0)
    feats = log_mel(wav_t, dither=0.0)
    # 4 feature frames (10 ms) per encoder frame (40 ms).
    n_enc = int(lens[0])
    feat_energy = feats.pow(2).mean(-1).cpu().numpy()
    energy = np.zeros(n_enc, dtype=np.float64)
    for t in range(n_enc):
        a, b = t * 4, min(feat_energy.shape[0], t * 4 + 4)
        if a < feat_energy.shape[0]:
            energy[t] = float(feat_energy[a:b].mean())
    energy_med = float(np.median(energy[energy > 0])) if np.any(energy > 0) else 1.0

    rnnt_emit = np.zeros(n_enc, dtype=np.int8)
    for t in timed.timestamps[0]:
        if 0 <= t < n_enc:
            rnnt_emit[t] = 1

    hyp_n = [geez_normalize(w["text"], fold_geez=True) for w in hyp_words]
    ref_n = [w["text"] for w in ref_words]
    hyp_keep = [(i, w) for i, w in enumerate(hyp_n) if w]
    ref_keep = [(i, w) for i, w in enumerate(ref_n) if w]
    ops = _align([w for _, w in ref_keep], [w for _, w in hyp_keep])

    events = []
    ri = hi = 0
    for kind, rw, hw, extra in ops:
        r_idx = [ref_keep[ri + k][0] for k in range(len(rw))]
        h_idx = [hyp_keep[hi + k][0] for k in range(len(hw))]
        ri += len(rw)
        hi += len(hw)
        if kind == "equal":
            continue
        spans = []
        for i in r_idx:
            spans.append((ref_words[i]["start_s"], ref_words[i]["end_s"]))
        for i in h_idx:
            spans.append((hyp_words[i]["start_s"], hyp_words[i]["end_s"]))
        if not spans:
            continue
        t0, t1 = min(s for s, _ in spans), max(e for _, e in spans)
        sl = _window(blank_p, t0, t1, FRAME_S)
        if sl.stop <= sl.start:
            continue
        b = blank_p[sl]
        e = energy[sl]
        H = entropy[sl]
        l2 = enc_l2[sl]
        emit = rnnt_emit[sl]
        idx = np.arange(sl.start, sl.stop)
        edge = np.minimum(idx % CHUNK_FRAMES, CHUNK_FRAMES - (idx % CHUNK_FRAMES))
        loud = float(e.mean()) >= 0.4 * energy_med
        events.append(
            {
                "kind": kind,
                "ref": " ".join(rw),
                "hyp": " ".join(hw),
                "ed": extra,
                "start_s": round(t0, 3),
                "end_s": round(t1, 3),
                "n_frames": int(sl.stop - sl.start),
                "blank_p": round(float(b.mean()), 3),
                "entropy": round(float(H.mean()), 3),
                "top_p": round(float(top_p[sl].mean()), 3),
                "energy": round(float(e.mean()), 5),
                "energy_vs_med": round(float(e.mean()) / max(energy_med, 1e-8), 2),
                "enc_l2": round(float(l2.mean()), 3),
                "rnnt_emit_frac": round(float(emit.mean()), 3),
                "min_chunk_edge": int(edge.min()),
                "at_chunk_edge": bool(int(edge.min()) <= 1),
                "loud": loud,
                "top_token": tokenizer.decode([int(top_i[sl.start + int(b.argmin())])], collapse=False),
            }
        )

    def _bucket(ev: dict) -> str:
        if ev["kind"] in ("drop", "drop_tail"):
            if not ev["loud"]:
                return "drop_quiet"
            if ev["at_chunk_edge"] and ev["blank_p"] >= 0.55:
                return "drop_chunk_edge"
            if ev["blank_p"] >= 0.7:
                return "drop_blanked"
            return "drop_weak"
        if ev["kind"] in ("wrong_word", "near_miss"):
            if ev["blank_p"] >= 0.6:
                return "sub_blank_then_lm"
            if ev["entropy"] >= 2.5:
                return "sub_uncertain"
            return "sub_confident_wrong"
        return ev["kind"]

    for ev in events:
        ev["bucket"] = _bucket(ev)

    counts: dict[str, int] = {}
    for ev in events:
        counts[ev["bucket"]] = counts.get(ev["bucket"], 0) + 1
        counts[ev["kind"]] = counts.get(ev["kind"], 0) + 1

    def _trace(t0: float, t1: float) -> list[dict]:
        pad = 0.25
        sl = _window(blank_p, t0 - pad, t1 + pad, FRAME_S)
        rows = []
        for i in range(sl.start, sl.stop):
            rows.append(
                {
                    "t": round(i * FRAME_S, 3),
                    "blank": round(float(blank_p[i]), 3),
                    "entropy": round(float(entropy[i]), 3),
                    "energy": round(float(energy[i]) / max(energy_med, 1e-8), 3),
                    "enc_l2": round(float(enc_l2[i]), 3),
                    "rnnt": int(rnnt_emit[i]),
                    "chunk": int(i % CHUNK_FRAMES == 0),
                }
            )
        return rows

    examples = {"drop": [], "wrong": []}
    for ev in events:
        if ev["kind"] in ("drop", "drop_tail") and ev["loud"] and len(examples["drop"]) < 4:
            ev = dict(ev)
            ev["trace"] = _trace(ev["start_s"], ev["end_s"])
            examples["drop"].append(ev)
        if ev["kind"] in ("wrong_word", "near_miss") and len(examples["wrong"]) < 4:
            ev = dict(ev)
            ev["trace"] = _trace(ev["start_s"], ev["end_s"])
            examples["wrong"].append(ev)

    result = {
        "wav": str(_resolve(args.wav)),
        "ref_srt": str(_resolve(args.ref_srt)),
        "ckpt": str(ckpt_path),
        "audio_s": round(duration, 3),
        "encoder_frames": n_enc,
        "frame_s": FRAME_S,
        "chunk_s": CHUNK_FRAMES * FRAME_S,
        "energy_median": round(energy_med, 5),
        "hyp_words": len(hyp_words),
        "ref_words": len(ref_words),
        "transcripts": {"ctc": ctc_text, "ctc_kenlm": kenlm_text},
        "counts": counts,
        "n_events": len(events),
        "events": events,
        "examples": examples,
        "global": {
            "mean_blank": round(float(blank_p.mean()), 3),
            "mean_entropy": round(float(entropy.mean()), 3),
            "mean_enc_l2": round(float(enc_l2.mean()), 3),
            "rnnt_emit_frac": round(float(rnnt_emit.mean()), 3),
            "loud_frame_frac": round(float((energy >= 0.4 * energy_med).mean()), 3),
        },
    }
    save_json(out, result)
    log.info("wrote %s  %d events  buckets %s", out, len(events), counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
