#!/usr/bin/env python3
"""Timed transcription of a YouTube clip.

Zipformer checkpoints use the demo ``/stream`` path (250 ms pumps into
``ZipformerStream``). Conformer CTC checkpoints use the demo upload path
(bandlimit, quiet-mic gain, 0.3 s + 0.5 s room-noise pad). Both rescore
CTC with the word 5-gram LM. Subtitles use pyctcdecode word frames.

    python eval_youtube_timed.py \\
        --url 'https://www.youtube.com/watch?v=YiB33Piqh20'

    python eval_youtube_timed.py \\
        --config configs/20m_char_12layer_pool.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import re
from pathlib import Path

import torch

from decoding.greedy import greedy_decode
from decoding.kenlm import (
    build_decoder,
    labels_from_tokenizer,
    load_unigrams,
)
from eval_zipformer_stream import (
    PUMP_MS,
    _find_lm,
    _load_wav,
    _resolve,
)
from evaluation.wer import score_texts
from models.features import HOP_MS, SAMPLE_RATE, log_mel
from train import load_cmvn, load_tokenizer
from utils import ROOT, load_yaml, save_json, setup_logging

log = setup_logging()

# Feature hop 10 ms, Conv2d embed subsample 2×, Zipformer output downsample 2×.
# One joiner / CTC frame is therefore 40 ms of audio.
FRAME_S = (HOP_MS / 1000.0) * 2 * 2
DEFAULT_URL = "https://www.youtube.com/watch?v=YiB33Piqh20"
GAP_S = 0.45
MAX_CUE_S = 6.0
MAX_CUE_WORDS = 12


def _srt_clock(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def download_audio(url: str, out_dir: Path, max_seconds: float) -> tuple[Path, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    probe = subprocess.run(
        [
            "yt-dlp",
            "--no-playlist",
            "--print",
            "%(.{id,title,duration,channel,webpage_url})j",
            url,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    info = json.loads(probe.stdout.strip().splitlines()[-1])
    vid = info["id"]
    wav_path = out_dir / f"{vid}.wav"
    if wav_path.exists():
        return wav_path, info
    for cached in ROOT.glob(f"exp/*/youtube_timed/{vid}.wav"):
        log.info("reusing wav %s", cached)
        return cached, info
    raw_tmpl = str(out_dir / f"{vid}.src.%(ext)s")
    log.info("downloading %s", info.get("title") or vid)
    subprocess.run(
        [
            "yt-dlp",
            "--no-playlist",
            "-f",
            "bestaudio/best",
            "-o",
            raw_tmpl,
            info.get("webpage_url") or url,
        ],
        check=True,
    )
    raw = next(out_dir.glob(f"{vid}.src.*"))
    ffmpeg = [
        "ffmpeg",
        "-y",
        "-i",
        str(raw),
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-vn",
    ]
    if max_seconds > 0:
        ffmpeg.extend(["-t", f"{max_seconds:.3f}"])
    ffmpeg.append(str(wav_path))
    subprocess.run(ffmpeg, check=True, capture_output=True)
    raw.unlink(missing_ok=True)
    return wav_path, info


_SRT_TS = re.compile(r"^\d+$")
_SRT_CLOCK = re.compile(r"\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->")
_TAG = re.compile(r"<[^>]+>")


def download_subs(url: str, out_dir: Path, vid: str) -> dict[str, Path]:
    """Manual Amharic first, then YouTube auto ``am`` / ``am-orig``."""
    found: dict[str, Path] = {}
    for kind, extra in (
        ("manual", ["--write-subs", "--sub-langs", "am"]),
        ("auto", ["--write-auto-subs", "--sub-langs", "am,am-orig"]),
    ):
        sub_dir = out_dir / "yt_subs"
        sub_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "yt-dlp",
                "--no-playlist",
                "--skip-download",
                "--convert-subs",
                "srt",
                "-o",
                str(sub_dir / vid),
                *extra,
                url,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        for path in sub_dir.glob(f"{vid}*"):
            if path.suffix.lower() != ".srt":
                continue
            name = path.name.lower()
            key = "am-orig" if "am-orig" in name else ("am" if ".am." in name or name.endswith(".am.srt") else None)
            if key:
                found[f"{kind}_{key}"] = path
                log.info("subs %s %s", kind, path.name)
    return found


def srt_text(path: Path) -> str:
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or _SRT_TS.match(line) or _SRT_CLOCK.match(line):
            continue
        line = _TAG.sub("", line).replace("&nbsp;", " ").strip()
        if line:
            lines.append(line)
    # YouTube auto-captions repeat rolling phrases; keep first occurrence of a run.
    out: list[str] = []
    prev = ""
    for line in lines:
        if line == prev:
            continue
        out.append(line)
        prev = line
    return " ".join(out)


def kenlm_words(
    decoder,
    logits,
    length: int,
    beam_width: int,
    *,
    frame_s: float,
    time_offset_s: float = 0.0,
) -> tuple[str, list[dict]]:
    """Best CTC+word-LM beam plus per-word start/end in seconds."""
    arr = logits.detach().float().cpu().numpy()[0, :length]
    beams = decoder.decode_beams(arr, beam_width=beam_width)
    if not beams:
        return "", []
    text, _state, word_frames, _logit, _lm = beams[0]
    words = []
    for word, (start, end) in word_frames:
        start_s = max(0.0, int(start) * frame_s - time_offset_s)
        end_s = max(start_s + frame_s, (int(end) + 1) * frame_s - time_offset_s)
        words.append(
            {
                "text": word,
                "start_s": round(start_s, 3),
                "end_s": round(end_s, 3),
            }
        )
    return text, words


def token_words(ids: list[int], frames: list[int], tokenizer) -> list[dict]:
    words: list[dict] = []
    buf_ids: list[int] = []
    buf_t: list[int] = []
    for tok, t in zip(ids, frames):
        piece = tokenizer.decode([tok], collapse=False)
        if piece.isspace():
            if buf_ids:
                text = tokenizer.decode(buf_ids, collapse=False)
                words.append(
                    {
                        "text": text,
                        "start_s": round(buf_t[0] * FRAME_S, 3),
                        "end_s": round((buf_t[-1] + 1) * FRAME_S, 3),
                    }
                )
                buf_ids, buf_t = [], []
            continue
        buf_ids.append(tok)
        buf_t.append(t)
    if buf_ids:
        text = tokenizer.decode(buf_ids, collapse=False)
        words.append(
            {
                "text": text,
                "start_s": round(buf_t[0] * FRAME_S, 3),
                "end_s": round((buf_t[-1] + 1) * FRAME_S, 3),
            }
        )
    return words


def group_cues(words: list[dict]) -> list[dict]:
    cues: list[dict] = []
    cur: list[dict] = []
    for w in words:
        if not cur:
            cur = [w]
            continue
        gap = w["start_s"] - cur[-1]["end_s"]
        dur = w["end_s"] - cur[0]["start_s"]
        if gap >= GAP_S or dur >= MAX_CUE_S or len(cur) >= MAX_CUE_WORDS:
            cues.append(_cue(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        cues.append(_cue(cur))
    return cues


def _cue(words: list[dict]) -> dict:
    return {
        "start_s": words[0]["start_s"],
        "end_s": max(words[-1]["end_s"], words[0]["start_s"] + 0.4),
        "text": " ".join(w["text"] for w in words),
        "n_words": len(words),
    }


def write_srt(path: Path, cues: list[dict]) -> None:
    blocks = []
    for i, cue in enumerate(cues, 1):
        blocks.append(
            f"{i}\n{_srt_clock(cue['start_s'])} --> {_srt_clock(cue['end_s'])}\n{cue['text']}\n"
        )
    path.write_text("\n".join(blocks), encoding="utf-8")


def _is_zipformer(cfg: dict) -> bool:
    return str(cfg.get("model", {}).get("name", "")).startswith("zipformer")


def decode_zipformer(model, tokenizer, cmvn, kenlm, wav, sr, device, args):
    from decoding.zipformer_greedy import greedy_search_batch
    from eval_zipformer_stream import stream_utterance

    sess = stream_utterance(model, cmvn, wav, sr, device, args.pump_ms)
    if sess.encoder_out is None:
        raise RuntimeError("encoder produced no frames")
    encoder_out = sess.encoder_out
    lens = torch.tensor([encoder_out.size(1)], device=device)
    timed = greedy_search_batch(model, encoder_out, lens, return_timestamps=True)
    rnnt_text = tokenizer.decode(timed.hyps[0], collapse=False)
    logits = model.ctc_output[1](encoder_out)
    ctc_text = greedy_decode(logits, tokenizer, lens)[0]
    kenlm_text, words = kenlm_words(
        kenlm, logits, int(lens[0]), args.beam_width, frame_s=FRAME_S
    )
    return {
        "mode": "demo_stream",
        "frame_s": FRAME_S,
        "encoder_frames": int(encoder_out.size(1)),
        "pump_ms": args.pump_ms,
        "pad_lead_s": 0.0,
        "rnnt": rnnt_text,
        "ctc": ctc_text,
        "ctc_kenlm": kenlm_text,
        "words": words,
    }


def decode_conformer(model, tokenizer, cmvn, kenlm, wav, sr, device, args):
    """Demo upload path. Long clips are windowed: rel-pos attention OOMs past ~90 s on 4 GB."""
    from demo.app import _pad_edges, _prep_waveform

    lead_s = 0.3
    chunk_s = 60.0
    overlap_s = 2.0
    wav_t = torch.from_numpy(wav.copy())
    if sr != SAMPLE_RATE:
        import torchaudio

        wav_t = torchaudio.functional.resample(wav_t.unsqueeze(0), sr, SAMPLE_RATE).squeeze(0)
    wav_t = _prep_waveform(wav_t)
    time_ratio = int(getattr(model.encoder, "time_ratio", 4))
    frame_s = (HOP_MS / 1000.0) * time_ratio
    n = int(wav_t.numel())
    win = int(chunk_s * SAMPLE_RATE)
    hop = int((chunk_s - overlap_s) * SAMPLE_RATE)
    offsets = [0] if n <= win else list(range(0, max(1, n - win + 1), hop))
    if offsets[-1] + win < n:
        offsets.append(max(0, n - win))

    ctc_parts: list[str] = []
    kenlm_parts: list[str] = []
    words: list[dict] = []
    n_frames = 0
    for start in offsets:
        end = min(start + win, n)
        audio = _pad_edges(wav_t[start:end])
        feats = log_mel(audio, dither=1e-5)
        if cmvn is not None:
            feats = cmvn(feats)
        feats = feats.unsqueeze(0).to(device)
        lens = torch.tensor([feats.size(1)], device=device, dtype=torch.long)
        logits, logit_lens = model(feats, lens)
        n_frames += int(logit_lens[0])
        ctc_parts.append(greedy_decode(logits, tokenizer, logit_lens)[0])
        text, chunk_words = kenlm_words(
            kenlm,
            logits,
            int(logit_lens[0]),
            args.beam_width,
            frame_s=frame_s,
            time_offset_s=lead_s,
        )
        kenlm_parts.append(text)
        t0 = start / SAMPLE_RATE
        for w in chunk_words:
            w["start_s"] = round(w["start_s"] + t0, 3)
            w["end_s"] = round(w["end_s"] + t0, 3)
        if words and start > 0:
            cut = t0 + overlap_s / 2
            words = [w for w in words if w["start_s"] < cut]
            chunk_words = [w for w in chunk_words if w["start_s"] >= cut]
        words.extend(chunk_words)
        del feats, logits
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "mode": "demo_upload" if len(offsets) == 1 else "demo_upload_chunked",
        "frame_s": frame_s,
        "encoder_frames": n_frames,
        "pump_ms": None,
        "pad_lead_s": lead_s,
        "rnnt": "",
        "ctc": " ".join(p for p in ctc_parts if p),
        "ctc_kenlm": " ".join(w["text"] for w in words) or " ".join(p for p in kenlm_parts if p),
        "words": words,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--config", default="configs/zipformer_s_streaming_pool.yaml")
    p.add_argument("--ckpt", default=None, help="default: <exp_dir>/best.pt")
    p.add_argument(
        "--max-seconds",
        type=float,
        default=180.0,
        help="trim downloaded audio; 0 keeps the full clip",
    )
    p.add_argument("--pump-ms", type=int, default=PUMP_MS)
    p.add_argument("--lm", default=None, help="word KenLM lm.bin or a folder containing it")
    p.add_argument("--lm-alpha", type=float, default=0.5)
    p.add_argument("--lm-beta", type=float, default=0.8)
    p.add_argument("--beam-width", type=int, default=100)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    cfg = load_yaml(_resolve(args.config))
    exp_dir = _resolve(cfg["paths"]["exp_dir"])
    ckpt_path = _resolve(args.ckpt) if args.ckpt else exp_dir / "best.pt"
    out_dir = _resolve(args.out_dir) if args.out_dir else exp_dir / "youtube_timed"
    out_dir.mkdir(parents=True, exist_ok=True)
    zipformer = _is_zipformer(cfg)

    wav_path, info = download_audio(args.url, out_dir, args.max_seconds)
    wav, sr = _load_wav(wav_path)
    duration = wav.shape[0] / max(sr, 1)
    log.info(
        "clip %s  %.1fs  %s  model=%s",
        info.get("id"),
        duration,
        info.get("title"),
        cfg["model"].get("name"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(cfg, exp_dir)
    cmvn = load_cmvn(cfg, exp_dir, compute_if_missing=False)
    if zipformer:
        from models.zipformer_icefall import build_zipformer, load_icefall

        load_icefall()
        model = build_zipformer(cfg, tokenizer.vocab_size)
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        model.to(device).eval()
        epoch = blob.get("epoch")
    else:
        from evaluate import load_model

        model = load_model(cfg, ckpt_path, tokenizer, device)
        epoch = torch.load(ckpt_path, map_location="cpu", weights_only=False).get("epoch")
    log.info("ckpt %s epoch=%s device=%s", ckpt_path, epoch, device)

    lm_path, uni_path = _find_lm(args.lm)
    unigrams = load_unigrams(uni_path)
    log.info(
        "word kenlm %s unigrams=%s (%d) alpha=%.2f beta=%.2f beam=%d",
        lm_path,
        uni_path,
        len(unigrams or []),
        args.lm_alpha,
        args.lm_beta,
        args.beam_width,
    )
    kenlm = build_decoder(
        labels_from_tokenizer(tokenizer),
        str(lm_path),
        unigrams=unigrams,
        alpha=args.lm_alpha,
        beta=args.lm_beta,
    )

    t0 = time.perf_counter()
    with torch.inference_mode():
        decoded = (
            decode_zipformer(model, tokenizer, cmvn, kenlm, wav, sr, device, args)
            if zipformer
            else decode_conformer(model, tokenizer, cmvn, kenlm, wav, sr, device, args)
        )
    wall = time.perf_counter() - t0

    words = decoded["words"]
    cues = group_cues(words)
    stem = out_dir / info["id"]
    write_srt(Path(str(stem) + ".srt"), cues)
    result = {
        "url": info.get("webpage_url") or args.url,
        "video_id": info.get("id"),
        "title": info.get("title"),
        "channel": info.get("channel"),
        "ckpt": str(ckpt_path),
        "epoch": epoch,
        "model": cfg["model"].get("name"),
        "mode": decoded["mode"],
        "source_duration_s": info.get("duration"),
        "audio_s": round(duration, 3),
        "encoder_frames": decoded["encoder_frames"],
        "frame_s": decoded["frame_s"],
        "pump_ms": decoded["pump_ms"],
        "pad_lead_s": decoded["pad_lead_s"],
        "wall_s": round(wall, 2),
        "rtf": round(wall / max(duration, 1e-6), 3),
        "n_words": len(words),
        "n_cues": len(cues),
        "lm": str(lm_path),
        "lm_alpha": args.lm_alpha,
        "lm_beta": args.lm_beta,
        "beam_width": args.beam_width,
        "transcripts": {
            "rnnt": decoded["rnnt"],
            "ctc": decoded["ctc"],
            "ctc_kenlm": decoded["ctc_kenlm"],
        },
        "words": words,
        "cues": cues,
    }
    save_json(Path(str(stem) + ".json"), result)
    lines = [
        f"# {info.get('title')}",
        f"# {info.get('webpage_url') or args.url}",
        f"# {ckpt_path}",
        "",
    ]
    if decoded["rnnt"]:
        lines.extend(["## RNN-T (timed)", decoded["rnnt"], ""])
    lines.extend(
        [
            "## CTC greedy",
            decoded["ctc"],
            "",
            "## CTC + word KenLM",
            decoded["ctc_kenlm"],
            "",
        ]
    )
    Path(str(stem) + ".txt").write_text("\n".join(lines), encoding="utf-8")
    vs_subs: dict[str, dict] = {}
    refs = download_subs(info.get("webpage_url") or args.url, out_dir, info["id"])
    for label, path in refs.items():
        ref = srt_text(path)
        if not ref.strip():
            continue
        vs_subs[label] = {
            "path": str(path),
            "n_chars": len(ref),
            "n_words": len(ref.split()),
            "scores": {
                name: score_texts([ref], [text], fold_geez=True)
                for name, text in decoded.items()
                if name in ("rnnt", "ctc", "ctc_kenlm") and text
            },
        }
        log.info("ref %s  %d words  %s", label, vs_subs[label]["n_words"], path.name)
        for name, sc in vs_subs[label]["scores"].items():
            log.info(
                "  vs %s  folded WER %.2f CER %.2f",
                name,
                sc["wer_pct"],
                sc["cer_pct"],
            )
        Path(str(stem) + f".ref_{label}.txt").write_text(ref, encoding="utf-8")
    if vs_subs:
        result["vs_youtube_subs"] = vs_subs
        save_json(Path(str(stem) + ".json"), result)
    log.info(
        "wrote %s.srt  %d cues  %d words  RTF %.2f",
        stem.name,
        len(cues),
        len(words),
        wall / max(duration, 1e-6),
    )
    for cue in cues[:8]:
        log.info("[%s → %s] %s", _srt_clock(cue["start_s"]), _srt_clock(cue["end_s"]), cue["text"])
    if len(cues) > 8:
        log.info("... %d more cues", len(cues) - 8)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
