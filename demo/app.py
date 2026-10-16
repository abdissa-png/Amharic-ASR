#!/usr/bin/env python3
"""Local mic demo (no Gradio, no CDN).

    python -m demo.app

Binds to all interfaces. Laptop: http://127.0.0.1:7860 (mic works).
Phone on the same Wi-Fi: https://<lan-ip>:7860 — accept the self-signed
cert once. Port 7860 speaks HTTP and HTTPS (Chrome tries HTTPS first).
"""

from __future__ import annotations

import argparse
import json
import socket
import ssl
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
import torchaudio

from decoding.greedy import greedy_decode
from decoding.unit_lm import decode_batch as unit_lm_decode_batch
from decoding.unit_lm import load_unit_model
from decoding.zipformer_greedy import greedy_search_batch
from decoding.zipformer_onnx import OnnxZipformer, OnnxZipformerStream
from decoding.zipformer_stream import EMBED_CHUNK, ZipformerStream, prune_streams
from decoding.kenlm import (
    build_decoder,
    decode_batch as kenlm_decode_batch,
    labels_from_tokenizer,
    lm_files,
    load_unigrams,
    resolve_lm_dir,
)
from evaluate import load_model
from models.features import SAMPLE_RATE, log_mel
from train import load_cmvn, load_tokenizer
from utils import ROOT, load_yaml, setup_logging

log = setup_logging()

STATIC = Path(__file__).resolve().parent / "static"
CERT_DIR = Path(__file__).resolve().parent / ".certs"

WORD_LM_DIRS = [
    ROOT / "exp" / "20m_char_12layer" / "lm_word",
    ROOT / "exp" / "lm_word",
]
CHAR_LM_DIR = ROOT / "exp" / "zipformer_s_streaming_pool" / "lm_char"
CHAR_LM_ALPHA = 0.4
CHAR_LM_BETA = 1.2
CHAR_LM_BEAM = 32
CHAR_LM_TOP_K = 12

PRESETS = {
    "zipformer_s_pool_leyu": {
        "label": "Zipformer-S streaming pool + Leyu (exp/zipformer_s_streaming_pool_leyu/last.pt)",
        "config": ROOT / "configs" / "zipformer_s_streaming_pool_leyu.yaml",
        "ckpt": ROOT / "exp" / "zipformer_s_streaming_pool_leyu" / "last.pt",
        "primary_label": "Transducer",
        "secondary_label": "CTC",
        "streaming": True,
        "lm_dirs": WORD_LM_DIRS,
        "char_lm_dir": CHAR_LM_DIR,
        "default_lm": "char",
    },
    "zipformer_s_pool_leyu_onnx": {
        "label": "Zipformer-S Leyu ONNX fp32 (chunk 32)",
        "config": ROOT / "configs" / "zipformer_s_streaming_pool_leyu.yaml",
        "ckpt": ROOT / "exp" / "zipformer_s_streaming_pool_leyu" / "onnx" / "encoder-chunk32-left128.onnx",
        "onnx_paths": (
            ROOT / "exp" / "zipformer_s_streaming_pool_leyu" / "onnx" / "encoder-chunk32-left128.onnx",
            ROOT / "exp" / "zipformer_s_streaming_pool_leyu" / "onnx" / "decoder-chunk32-left128.onnx",
            ROOT / "exp" / "zipformer_s_streaming_pool_leyu" / "onnx" / "joiner-chunk32-left128.onnx",
        ),
        "primary_label": "Transducer",
        "secondary_label": "CTC",
        "streaming": True,
        "lm_dirs": WORD_LM_DIRS,
        "char_lm_dir": CHAR_LM_DIR,
        "default_lm": "char",
    },
    "zipformer_s_pool_leyu_onnx_int8": {
        "label": "Zipformer-S Leyu ONNX int8 (chunk 32)",
        "config": ROOT / "configs" / "zipformer_s_streaming_pool_leyu.yaml",
        "ckpt": ROOT / "exp" / "zipformer_s_streaming_pool_leyu" / "onnx" / "encoder-chunk32-left128.int8.onnx",
        "onnx_paths": (
            ROOT / "exp" / "zipformer_s_streaming_pool_leyu" / "onnx" / "encoder-chunk32-left128.int8.onnx",
            ROOT / "exp" / "zipformer_s_streaming_pool_leyu" / "onnx" / "decoder-chunk32-left128.int8.onnx",
            ROOT / "exp" / "zipformer_s_streaming_pool_leyu" / "onnx" / "joiner-chunk32-left128.int8.onnx",
        ),
        "primary_label": "Transducer",
        "secondary_label": "CTC",
        "streaming": True,
        "lm_dirs": WORD_LM_DIRS,
        "char_lm_dir": CHAR_LM_DIR,
        "default_lm": "char",
    },
    "zipformer_s_pool": {
        "label": "Zipformer-S streaming pool (exp/zipformer_s_streaming_pool)",
        "config": ROOT / "configs" / "zipformer_s_streaming_pool.yaml",
        "ckpt": ROOT / "exp" / "zipformer_s_streaming_pool" / "best.pt",
        "primary_label": "Transducer",
        "secondary_label": "CTC",
        "streaming": True,
        "lm_dirs": WORD_LM_DIRS,
        "char_lm_dir": CHAR_LM_DIR,
        "default_lm": "char",
    },
    "zipformer_s_align": {
        "label": "Zipformer-S streaming align (exp/zipformer_s_streaming_et_align)",
        "config": ROOT / "configs" / "zipformer_s_streaming_et_align.yaml",
        "ckpt": ROOT / "exp" / "zipformer_s_streaming_et_align" / "best.pt",
        "primary_label": "Transducer",
        "secondary_label": "CTC",
        "streaming": True,
        "lm_dirs": WORD_LM_DIRS,
        "char_lm_dir": CHAR_LM_DIR,
        "default_lm": "none",
    },
    "char12_pool_short": {
        "label": "12-layer pooled char, short-crop fine-tune (exp/20m_char_12layer_pool_short)",
        "config": ROOT / "configs" / "20m_char_12layer_pool_short.yaml",
        "ckpt": ROOT / "exp" / "20m_char_12layer_pool_short" / "best.pt",
        "lm_dirs": [
            ROOT / "exp" / "lm_word",
            ROOT / "exp" / "20m_char_12layer_pool" / "lm_word",
            ROOT / "exp" / "20m_char_12layer" / "lm_word",
            ROOT / "exp" / "20m_char_12layer" / "lm",
        ],
    },
    "char12_pool": {
        "label": "12-layer pooled char + word LM (exp/20m_char_12layer_pool)",
        "config": ROOT / "configs" / "20m_char_12layer_pool.yaml",
        "ckpt": ROOT / "exp" / "20m_char_12layer_pool" / "best.pt",
        "lm_dirs": [
            ROOT / "exp" / "lm_word",
            ROOT / "exp" / "20m_char_12layer_pool" / "lm_word",
            ROOT / "exp" / "20m_char_12layer" / "lm_word",
            ROOT / "exp" / "20m_char_12layer" / "lm",
        ],
    },
    "char12_pool_onnx": {
        "label": "12-layer pooled char ONNX fp32 (exp/20m_char_12layer_pool)",
        "config": ROOT / "configs" / "20m_char_12layer_pool.yaml",
        "ckpt": ROOT / "exp" / "20m_char_12layer_pool" / "model.onnx",
        "onnx_ctc": ROOT / "exp" / "20m_char_12layer_pool" / "model.onnx",
        "lm_dirs": [
            ROOT / "exp" / "lm_word",
            ROOT / "exp" / "20m_char_12layer_pool" / "lm_word",
            ROOT / "exp" / "20m_char_12layer" / "lm_word",
            ROOT / "exp" / "20m_char_12layer" / "lm",
        ],
    },
    "char12_pool_onnx_int8": {
        "label": "12-layer pooled char ONNX int8 (exp/20m_char_12layer_pool)",
        "config": ROOT / "configs" / "20m_char_12layer_pool.yaml",
        "ckpt": ROOT / "exp" / "20m_char_12layer_pool" / "model.int8.onnx",
        "onnx_ctc": ROOT / "exp" / "20m_char_12layer_pool" / "model.int8.onnx",
        "lm_dirs": [
            ROOT / "exp" / "lm_word",
            ROOT / "exp" / "20m_char_12layer_pool" / "lm_word",
            ROOT / "exp" / "20m_char_12layer" / "lm_word",
            ROOT / "exp" / "20m_char_12layer" / "lm",
        ],
    },
    "char12": {
        "label": "12-layer char + word LM (exp/20m_char_12layer)",
        "config": ROOT / "configs" / "20m_char_12layer.yaml",
        "ckpt": ROOT / "exp" / "20m_char_12layer" / "best.pt",
        # Prefer word LM when present; fall back to speech-only.
        "lm_dirs": [
            ROOT / "exp" / "lm_word",
            ROOT / "exp" / "20m_char_12layer" / "lm_word",
            ROOT / "exp" / "20m_char_12layer" / "lm",
        ],
    },
    "char12_onnx": {
        "label": "12-layer char ONNX fp32 (exp/20m_char_12layer)",
        "config": ROOT / "configs" / "20m_char_12layer.yaml",
        "ckpt": ROOT / "exp" / "20m_char_12layer" / "model.onnx",
        "onnx_ctc": ROOT / "exp" / "20m_char_12layer" / "model.onnx",
        "lm_dirs": [
            ROOT / "exp" / "lm_word",
            ROOT / "exp" / "20m_char_12layer" / "lm_word",
            ROOT / "exp" / "20m_char_12layer" / "lm",
        ],
    },
    "char12_onnx_int8": {
        "label": "12-layer char ONNX int8 (exp/20m_char_12layer)",
        "config": ROOT / "configs" / "20m_char_12layer.yaml",
        "ckpt": ROOT / "exp" / "20m_char_12layer" / "model.int8.onnx",
        "onnx_ctc": ROOT / "exp" / "20m_char_12layer" / "model.int8.onnx",
        "lm_dirs": [
            ROOT / "exp" / "lm_word",
            ROOT / "exp" / "20m_char_12layer" / "lm_word",
            ROOT / "exp" / "20m_char_12layer" / "lm",
        ],
    },
    "char": {
        "label": "4-layer char (exp/20m_char_4layer)",
        "config": ROOT / "configs" / "20m_char_4layer.yaml",
        "ckpt": ROOT / "exp" / "20m_char_4layer" / "best.pt",
    },
    "bpe": {
        "label": "12-layer unigram BPE (exp/20m_bpe)",
        "config": ROOT / "configs" / "20m_bpe.yaml",
        "ckpt": ROOT / "exp" / "20m_bpe" / "best.pt",
    },
}


# Offline file upload uses the training-time chunk mask. 32 and 128 are one
# of the (chunk, left-context) pairs the causal encoder was trained on.
# Live /stream does not use this mask; it keeps the icefall streaming cache
# (decoding.zipformer_stream) and only emits a full embed chunk.
STREAM_CHUNK = EMBED_CHUNK
STREAM_LEFT = 128


def _lm_from_dirs(exp_dir: Path, extra: list[Path] | None) -> tuple[Path | None, Path | None]:
    for d in extra or []:
        model, uni = lm_files(d)
        if model is not None:
            return model, uni
    return resolve_lm_dir(exp_dir)


def _lms_for_spec(spec: dict) -> list[str]:
    out = ["none"]
    for d in spec.get("lm_dirs") or []:
        if lm_files(d)[0] is not None:
            out.append("word")
            break
    char_dir = spec.get("char_lm_dir")
    if char_dir is not None and lm_files(Path(char_dir))[0] is not None:
        out.append("char")
    return out


def _default_lm(spec: dict, lms: list[str]) -> str:
    if "default_lm" in spec:
        wanted = spec["default_lm"]
    elif "word" in lms:
        wanted = "word"
    else:
        wanted = "none"
    return wanted if wanted in lms else (lms[-1] if lms else "none")


def _lm_box_label(mode: str, *, ran: bool) -> str:
    if ran and mode == "char":
        return "CTC + char LM"
    if ran and mode == "word":
        return "CTC + word LM"
    return "CTC"


def _to_mono16k(sr: int, wav: np.ndarray) -> torch.Tensor:
    x = np.asarray(wav)
    if x.ndim == 2:
        ch_axis = 0 if x.shape[0] <= 8 else 1
        x = x.mean(axis=ch_axis)
    t = torch.from_numpy(np.ascontiguousarray(x)).float().reshape(-1)
    if t.abs().max() > 1.5:
        t = t / 32768.0
    t = t.clamp(-1.0, 1.0)
    if int(sr) != SAMPLE_RATE:
        t = torchaudio.functional.resample(
            t.unsqueeze(0),
            int(sr),
            SAMPLE_RATE,
            lowpass_filter_width=64,
            rolloff=0.9475,
            resampling_method="sinc_interp_kaiser",
        ).squeeze(0)
    return t


def _speech_rms(wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
    hop = max(1, sample_rate // 50)
    if wav.numel() < hop * 4:
        return wav.pow(2).mean().sqrt()
    frames = wav.unfold(0, hop, hop)
    energy = frames.pow(2).mean(-1)
    keep = energy >= energy.median()
    if not bool(keep.any()):
        return wav.pow(2).mean().sqrt()
    return frames[keep].pow(2).mean().sqrt()


def _noise_floor(wav: torch.Tensor, n: int, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """``n`` samples built from the clip's quietest 10 ms frames (its room noise)."""
    hop = max(1, sample_rate // 100)
    frames = wav[: wav.numel() // hop * hop].reshape(-1, hop)
    if n <= 0 or frames.size(0) < 4:
        return wav.new_zeros(max(0, n))
    quiet = frames[frames.pow(2).mean(-1).argsort()[: max(4, frames.size(0) // 10)]].reshape(-1)
    return quiet.repeat(n // quiet.numel() + 1)[:n]


def _pad_edges(
    wav: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    lead_s: float = 0.3,
    tail_s: float = 0.5,
) -> torch.Tensor:
    """Surround the clip with its own room noise.

    Training clips nearly always have silence after the last word. On FLEURS
    test, cutting clips at the end of speech dropped last-word accuracy from
    87% to 81%, and cutting 0.1 s early to 69%; 0.5 s of room noise restored 86%.
    Masked padding gives no context, so it has to be real audio.
    """
    lead = _noise_floor(wav, int(lead_s * sample_rate), sample_rate)
    tail = _noise_floor(wav, int(tail_s * sample_rate), sample_rate)
    return torch.cat([lead, wav, tail])


def _prep_waveform(wav: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Match Dataset.ET 16 kHz speech band. Gain only if the mic is too quiet.

    Training did not denoise, loudness-normalize, or trim silence.
    """
    if wav.numel() == 0:
        return wav
    wav = wav - wav.mean()
    if wav.numel() > sample_rate // 20:
        wav = torchaudio.functional.highpass_biquad(
            wav.unsqueeze(0), sample_rate, cutoff_freq=80.0
        ).squeeze(0)
        wav = torchaudio.functional.lowpass_biquad(
            wav.unsqueeze(0), sample_rate, cutoff_freq=7600.0
        ).squeeze(0)
    rms = _speech_rms(wav, sample_rate).clamp_min(1e-8)
    if float(rms) < 0.02:
        wav = wav * (0.06 / rms).clamp(max=8.0)
    peak = wav.abs().max()
    if float(peak) > 0.99:
        wav = wav * (0.99 / peak)
    return wav.clamp(-1.0, 1.0)


class LiveAsr:
    def __init__(
        self,
        config: Path,
        ckpt: Path,
        *,
        lm_alpha: float = 0.5,
        lm_beta: float = 0.8,
        beam_width: int = 100,
        lm_dirs: list[Path] | None = None,
        char_lm_dir: Path | None = None,
        char_lm_alpha: float = CHAR_LM_ALPHA,
        char_lm_beta: float = CHAR_LM_BETA,
        onnx_paths: tuple[Path, Path, Path] | None = None,
        onnx_ctc: Path | None = None,
    ) -> None:
        cfg = load_yaml(config)
        exp_dir = ROOT / cfg["paths"]["exp_dir"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = load_tokenizer(cfg, exp_dir)
        self.kind = "zipformer" if str(cfg["model"].get("name", "")).startswith("zipformer") else "ctc"
        self.primary_label = "Transducer" if self.kind == "zipformer" else "Greedy"
        self.secondary_label = "CTC" if self.kind == "zipformer" else "KenLM"
        self.onnx: OnnxZipformer | None = None
        self.onnx_ctc = None
        if onnx_paths is not None:
            self.kind = "zipformer"
            self.onnx = OnnxZipformer(
                *onnx_paths,
                blank_id=self.tokenizer.blank_id,
                unk_id=self.tokenizer.unk_id,
            )
            self.model = None
            self.ckpt_epoch = self.onnx.epoch
            if self.onnx.provider.startswith("CUDA"):
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        elif onnx_ctc is not None:
            from deployment.onnx_infer import OnnxCTC

            prefer = "cuda" if torch.cuda.is_available() else "cpu"
            self.onnx_ctc = OnnxCTC(onnx_ctc, prefer=prefer)
            self.model = self.onnx_ctc
            self.ckpt_epoch = None
            provider = self.onnx_ctc.providers[0] if self.onnx_ctc.providers else ""
            self.device = torch.device("cuda" if str(provider).startswith("CUDA") else "cpu")
            log.info("onnx ctc %s  providers=%s", onnx_ctc, ",".join(self.onnx_ctc.providers))
        elif self.kind == "zipformer":
            from models.zipformer_icefall import build_zipformer

            self.model = build_zipformer(cfg, self.tokenizer.vocab_size)
            blob = torch.load(ckpt, map_location=self.device, weights_only=False)
            self.model.load_state_dict(blob["model"])
            self.model.to(self.device).eval()
            self.ckpt_epoch = blob.get("epoch")
        else:
            self.model = load_model(cfg, ckpt, self.tokenizer, self.device)
            self.ckpt_epoch = None
        self.cmvn = load_cmvn(cfg, exp_dir, compute_if_missing=False)
        self.beam_width = beam_width
        self.char_lm_alpha = char_lm_alpha
        self.char_lm_beta = char_lm_beta
        self._lock = threading.Lock()
        self._streams: dict[str, ZipformerStream | OnnxZipformerStream] = {}
        self.kenlm = None
        self.char_km = None
        self.char_model = None
        notes: list[str] = []
        lm_path, uni_path = _lm_from_dirs(exp_dir, lm_dirs)
        if lm_path is None:
            searched = list(lm_dirs or []) + [exp_dir / "lm"]
            notes.append("no word LM in " + ", ".join(str(p) for p in searched))
        else:
            try:
                self.kenlm = build_decoder(
                    labels_from_tokenizer(self.tokenizer),
                    str(lm_path),
                    unigrams=load_unigrams(uni_path),
                    alpha=lm_alpha,
                    beta=lm_beta,
                )
                notes.append(f"word {lm_path.name}")
                log.info("word KenLM %s  alpha=%.2f beta=%.2f", lm_path, lm_alpha, lm_beta)
            except Exception as exc:
                notes.append(f"word LM failed: {type(exc).__name__}: {exc}")
                log.warning("%s", notes[-1], exc_info=True)
        if char_lm_dir is not None and lm_files(Path(char_lm_dir))[0] is not None:
            try:
                self.char_km, self.char_model, char_path = load_unit_model(char_lm_dir)
                notes.append(f"char {Path(char_path).name}")
                log.info(
                    "char KenLM %s  alpha=%.2f beta=%.2f beam=%d",
                    char_path,
                    char_lm_alpha,
                    char_lm_beta,
                    CHAR_LM_BEAM,
                )
            except Exception as exc:
                notes.append(f"char LM failed: {type(exc).__name__}: {exc}")
                log.warning("%s", notes[-1], exc_info=True)
        self.lm_note = "; ".join(notes) if notes else "CTC greedy"
        epoch_bit = f"epoch {self.ckpt_epoch}" if self.ckpt_epoch else "checkpoint"
        where = self.onnx.encoder_path if self.onnx is not None else ckpt
        log.info("ready  %s  %s  %s  device=%s  lm=%s", config, where, epoch_bit, self.device, self.lm_note)

    @torch.inference_mode()
    def transcribe(self, audio, *, lm_mode: str = "none", run_lm: bool = True) -> tuple[str, str, str, str]:
        if audio is None:
            return "", "", "Record or upload a clip first.", self.secondary_label
        if isinstance(audio, (str, Path)):
            import soundfile as sf

            wav, sr = sf.read(str(audio), dtype="float32", always_2d=False)
            audio = (sr, wav)
        if not (isinstance(audio, tuple) and len(audio) == 2):
            return "", "", f"Unexpected audio payload: {type(audio)}", self.secondary_label
        sr, wav = audio
        wav_t = _prep_waveform(_to_mono16k(int(sr), wav))
        return self._decode_waveform(wav_t, pad=True, lm_mode=lm_mode, run_lm=run_lm)

    @torch.inference_mode()
    def stream_pcm(
        self,
        sid: str,
        samples: np.ndarray,
        sample_rate: int,
        offset: int,
        *,
        lm_mode: str = "none",
        final: bool = False,
    ) -> tuple[str, str, str, str, int]:
        """Append new mic audio and decode only chunks the cache has not seen.

        ``offset`` is the sample index of ``samples[0]`` in the recording.
        Audio already accepted is skipped, so a retried POST does not
        re-encode. ``final`` flushes the tail once and may run the CTC LM.
        The last value is how many native samples this session has kept.
        """
        if self.kind != "zipformer":
            raise ValueError("This checkpoint decodes after you stop.")
        if not sid:
            raise ValueError("missing stream session")
        t0 = time.perf_counter()
        with self._lock:
            prune_streams(self._streams)
            sess = self._streams.get(sid)
            if sess is None:
                if int(offset) > 0:
                    raise ValueError("stream session expired; start recording again")
                sess = self._new_stream()
                self._streams[sid] = sess
            kept = sess.add(samples, sample_rate, offset, final=final)
            greedy = self.tokenizer.decode(sess.tokens, collapse=False) if sess.tokens else ""
            used_lm = "none"
            heard = sess.encoder_out is not None
            if not heard:
                secondary = ""
            else:
                secondary, used_lm = self._secondary_text(
                    sess.encoder_out,
                    torch.tensor([sess.encoder_out.size(1)]),
                    lm_mode=lm_mode,
                    run_lm=final,
                    ctc_logits=getattr(sess, "ctc_logits", None),
                )
        wall = time.perf_counter() - t0
        seconds = kept / max(1, int(sample_rate))
        if not heard:
            status = f"{seconds:.1f}s audio  listening"
        else:
            lm_bit = f"{'onnx' if self.onnx is not None else 'stream cache'} chunk {STREAM_CHUNK}"
            if used_lm == "char":
                lm_bit += "  char LM"
            elif used_lm == "word":
                lm_bit += "  word LM"
            else:
                lm_bit += "  CTC greedy"
            status = f"{seconds:.1f}s audio  {wall * 1000:.0f}ms  {self.device.type}  {lm_bit}"
        box_label = _lm_box_label(used_lm, ran=used_lm != "none")
        if used_lm == "none":
            box_label = self.secondary_label
        return greedy, secondary, status, box_label, kept

    def _new_stream(self) -> ZipformerStream | OnnxZipformerStream:
        if self.onnx is not None:
            return OnnxZipformerStream(self.onnx, self.cmvn, self.device)
        return ZipformerStream(self.model, self.cmvn, self.device)

    def _secondary_text(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        *,
        lm_mode: str,
        run_lm: bool,
        ctc_logits: torch.Tensor | None = None,
    ) -> tuple[str, str]:
        logits = ctc_logits if ctc_logits is not None else self.model.ctc_output[1](encoder_out)
        encoder_out_lens = encoder_out_lens.to(logits.device)
        text = greedy_decode(logits, self.tokenizer, encoder_out_lens)[0]
        used_lm = "none"
        if run_lm and lm_mode == "char" and self.char_model is not None:
            text = unit_lm_decode_batch(
                logits,
                encoder_out_lens,
                self.tokenizer,
                self.char_km,
                self.char_model,
                unit="char",
                beam_size=CHAR_LM_BEAM,
                token_top_k=CHAR_LM_TOP_K,
                alpha=self.char_lm_alpha,
                beta=self.char_lm_beta,
            )[0]
            used_lm = "char"
        elif run_lm and lm_mode == "word" and self.kenlm is not None:
            text = kenlm_decode_batch(
                self.kenlm, logits, encoder_out_lens, beam_width=self.beam_width
            )[0]
            used_lm = "word"
        return text, used_lm

    def _onnx_file(self, audio: torch.Tensor, seconds: float, *, lm_mode: str, run_lm: bool) -> tuple[str, str, str, str]:
        t0 = time.perf_counter()
        with self._lock:
            sess = self._new_stream()
            sess.run_prepared(audio)
            greedy = self.tokenizer.decode(sess.tokens, collapse=False) if sess.tokens else ""
            if sess.ctc_logits is None:
                kenlm_text, used_lm = "", "none"
            else:
                kenlm_text, used_lm = self._secondary_text(
                    sess.encoder_out,
                    torch.tensor([sess.ctc_logits.size(1)]),
                    lm_mode=lm_mode,
                    run_lm=run_lm,
                    ctc_logits=sess.ctc_logits,
                )
        wall = time.perf_counter() - t0
        box_label = _lm_box_label(used_lm, ran=used_lm != "none")
        if used_lm == "none":
            box_label = self.secondary_label
        status = f"{seconds:.1f}s audio  {wall * 1000:.0f}ms  {self.device.type}  onnx chunk {STREAM_CHUNK}"
        if used_lm == "char":
            status += "  char LM"
        elif used_lm == "word":
            status += "  word LM"
        else:
            status += "  CTC greedy"
        return greedy, kenlm_text, status, box_label

    @torch.inference_mode()
    def _decode_waveform(
        self, wav_t: torch.Tensor, *, pad: bool, lm_mode: str = "none", run_lm: bool = True
    ) -> tuple[str, str, str, str]:
        seconds = float(wav_t.numel()) / SAMPLE_RATE
        if wav_t.numel() < int(0.025 * SAMPLE_RATE):
            return "", "", f"Clip is {seconds:.2f}s; too short to score.", self.secondary_label
        audio = _pad_edges(wav_t) if pad else wav_t
        if self.onnx is not None:
            return self._onnx_file(audio, seconds, lm_mode=lm_mode, run_lm=run_lm)
        # dither=0 keeps a growing recording from changing frames that were already decoded.
        feats = log_mel(audio, dither=0.0 if self.kind == "zipformer" else 1e-5)
        if feats.numel() == 0:
            return "", "", f"Clip is {seconds:.2f}s; too short to score.", self.secondary_label
        if self.cmvn is not None:
            feats = self.cmvn(feats)
        feats = feats.unsqueeze(0).to(self.device)
        lens = torch.tensor([feats.size(1)], device=self.device, dtype=torch.long)
        used_lm = "none"
        t0 = time.perf_counter()
        with self._lock:
            if self.kind == "zipformer":
                greedy, kenlm_text, used_lm = self._zipformer_decode(
                    feats, lens, lm_mode=lm_mode, run_lm=run_lm
                )
            else:
                logits, logit_lens = self.model(feats, lens)
                greedy = greedy_decode(logits, self.tokenizer, logit_lens)[0]
                kenlm_text = ""
                mode = lm_mode
                if mode == "none" and self.kenlm is not None:
                    mode = "word"
                if run_lm and mode == "char" and self.char_model is not None:
                    kenlm_text = unit_lm_decode_batch(
                        logits,
                        logit_lens,
                        self.tokenizer,
                        self.char_km,
                        self.char_model,
                        unit="char",
                        beam_size=CHAR_LM_BEAM,
                        token_top_k=CHAR_LM_TOP_K,
                        alpha=self.char_lm_alpha,
                        beta=self.char_lm_beta,
                    )[0]
                    used_lm = "char"
                elif run_lm and mode != "none" and self.kenlm is not None:
                    try:
                        kenlm_text = kenlm_decode_batch(
                            self.kenlm, logits, logit_lens, beam_width=self.beam_width
                        )[0]
                        used_lm = "word"
                    except Exception as exc:
                        kenlm_text = f"(KenLM decode error: {exc})"
        wall = time.perf_counter() - t0
        box_label = _lm_box_label(used_lm, ran=used_lm != "none")
        if self.kind == "zipformer" and used_lm == "none":
            box_label = self.secondary_label
        if self.kind == "zipformer":
            lm_bit = f"chunk {STREAM_CHUNK}"
            if used_lm == "char":
                lm_bit += "  char LM"
            elif used_lm == "word":
                lm_bit += "  word LM"
            else:
                lm_bit += "  CTC greedy"
        else:
            lm_bit = {"char": "char LM", "word": "word LM"}.get(used_lm, self.lm_note)
            if self.onnx_ctc is not None:
                lm_bit = "onnx  " + lm_bit
        status = f"{seconds:.1f}s audio  {wall * 1000:.0f}ms  {self.device.type}  {lm_bit}"
        return greedy, kenlm_text, status, box_label

    def _zipformer_decode(
        self, feats: torch.Tensor, lens: torch.Tensor, *, lm_mode: str = "none", run_lm: bool = True
    ) -> tuple[str, str, str]:
        """Causal forward: each frame sees the current chunk and STREAM_LEFT frames of the past."""
        enc = self.model.encoder
        prev_chunk, prev_left = enc.chunk_size, enc.left_context_frames
        enc.chunk_size = (STREAM_CHUNK,)
        enc.left_context_frames = (STREAM_LEFT,)
        used_lm = "none"
        try:
            encoder_out, encoder_out_lens = self.model.forward_encoder(feats, lens)
            ids = greedy_search_batch(self.model, encoder_out, encoder_out_lens)[0]
            transducer = self.tokenizer.decode(ids, collapse=False)
            secondary, used_lm = self._secondary_text(
                encoder_out, encoder_out_lens, lm_mode=lm_mode, run_lm=run_lm
            )
        finally:
            enc.chunk_size = prev_chunk
            enc.left_context_frames = prev_left
        return transducer, secondary, used_lm

    def transcribe_bytes(
        self, data: bytes, *, lm_mode: str = "none", run_lm: bool = True
    ) -> tuple[str, str, str, str]:
        import io
        import tempfile

        import soundfile as sf

        try:
            wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
            return self.transcribe((sr, wav), lm_mode=lm_mode, run_lm=run_lm)
        except Exception:
            path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as fh:
                    fh.write(data)
                    path = fh.name
                wav_t, sr = torchaudio.load(path)
                return self.transcribe((int(sr), wav_t.mean(0).numpy()), lm_mode=lm_mode, run_lm=run_lm)
            except Exception as exc:
                return "", "", f"Could not decode audio ({exc})", self.secondary_label
            finally:
                if path:
                    Path(path).unlink(missing_ok=True)


def _lan_ipv4() -> str | None:
    """Best-effort primary LAN address (not 127.0.0.1)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
        if ip and not ip.startswith("127."):
            return ip
    except OSError:
        pass
    return None


def _ensure_tls(lan_ip: str | None) -> tuple[Path, Path]:
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    cert, key = CERT_DIR / "demo.crt", CERT_DIR / "demo.key"
    parts = ["DNS:localhost", "IP:127.0.0.1"]
    if lan_ip:
        parts.append(f"IP:{lan_ip}")
    san = "subjectAltName=" + ",".join(parts)
    stamp = CERT_DIR / "san.txt"
    if cert.exists() and key.exists() and stamp.exists() and stamp.read_text(encoding="utf-8") == san:
        return cert, key
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-sha256",
            "-days",
            "825",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-subj",
            "/CN=asr-demo",
            "-addext",
            san,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stamp.write_text(san, encoding="utf-8")
    log.info("wrote self-signed TLS cert %s (%s)", cert, san)
    return cert, key


def _ssl_context(cert: Path, key: Path) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(cert), str(key))
    return ctx


class DualStackHTTPServer(ThreadingHTTPServer):
    """HTTP on the socket; wrap in TLS when the first byte is a ClientHello (0x16).

    Chrome HTTPS-First sends TLS to port 7860 even if you typed http://.
    """

    def __init__(self, server_address, handler, ssl_context: ssl.SSLContext | None = None):
        super().__init__(server_address, handler)
        self.ssl_context = ssl_context

    def get_request(self):
        sock, addr = super().get_request()
        if self.ssl_context is None:
            return sock, addr
        sock.settimeout(5.0)
        try:
            peek = sock.recv(1, socket.MSG_PEEK)
        except OSError:
            sock.close()
            raise
        sock.settimeout(None)
        if peek[:1] == b"\x16":
            sock = self.ssl_context.wrap_socket(sock, server_side=True)
        return sock, addr


def _host_name(handler: BaseHTTPRequestHandler) -> str:
    return (handler.headers.get("Host") or "").split(":")[0].strip()


def _index_bytes(https_url: str | None) -> bytes:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    boot = f"window.ASR_HTTPS = {json.dumps(https_url)};"
    return html.replace("/*ASR_BOOT*/", boot).encode("utf-8")


def _preset_ready(spec: dict) -> bool:
    if not spec["config"].exists():
        return False
    paths = spec.get("onnx_paths")
    if paths:
        return all(Path(path).exists() for path in paths)
    onnx_ctc = spec.get("onnx_ctc")
    if onnx_ctc is not None:
        return Path(onnx_ctc).exists()
    return spec["ckpt"].exists()


def _available_presets() -> dict[str, dict]:
    return {key: spec for key, spec in PRESETS.items() if _preset_ready(spec)}


def _json(handler: BaseHTTPRequestHandler, code: int, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(
    get_engine,
    specs: dict,
    default: str,
    *,
    https_url: str | None = None,
    redirect_lan_to_https: bool = False,
):
    local_hosts = {"127.0.0.1", "localhost", "::1", ""}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            log.info("%s", fmt % args)

        def _is_tls(self) -> bool:
            return isinstance(getattr(self, "connection", None), ssl.SSLSocket)

        def _redirect_https(self) -> bool:
            if not (redirect_lan_to_https and https_url) or self._is_tls():
                return False
            host = _host_name(self)
            if host in local_hosts:
                return False
            loc = https_url.rstrip("/") + (self.path or "/")
            self.send_response(302)
            self.send_header("Location", loc)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return True

        def do_GET(self) -> None:
            if self._redirect_https():
                return
            path = urlparse(self.path).path
            if path == "/presets":
                listed = []
                for k in specs:
                    lms = _lms_for_spec(specs[k])
                    listed.append(
                        {
                            "id": k,
                            "label": specs[k]["label"],
                            "primary_label": specs[k].get("primary_label", "Greedy"),
                            "secondary_label": specs[k].get("secondary_label", "KenLM"),
                            "streaming": bool(specs[k].get("streaming")),
                            "lms": lms,
                            "default_lm": _default_lm(specs[k], lms),
                        }
                    )
                _json(self, 200, {"default": default, "presets": listed})
                return
            if path in ("/", "/index.html"):
                data = _index_bytes(https_url)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(404)

        def do_POST(self) -> None:
            if self._redirect_https():
                return
            path = urlparse(self.path)
            if path.path not in ("/transcribe", "/stream"):
                self.send_error(404)
                return
            n = int(self.headers.get("Content-Length") or 0)
            data = self.rfile.read(n) if n else b""
            qs = parse_qs(path.query)
            preset = (qs.get("preset") or [default])[0]
            lm_mode = (qs.get("lm") or ["none"])[0]
            if lm_mode not in ("none", "word", "char"):
                lm_mode = "none"
            final = (qs.get("final") or ["0"])[0].lower() in ("1", "true", "yes")
            engine = get_engine(preset)
            accepted = None
            if path.path == "/stream":
                if engine.kind != "zipformer":
                    _json(self, 400, {"error": "This checkpoint decodes after you stop."})
                    return
                sr = int((qs.get("sr") or ["16000"])[0])
                sid = (qs.get("sid") or [""])[0]
                try:
                    offset = int((qs.get("offset") or ["0"])[0])
                except ValueError:
                    offset = 0
                samples = np.frombuffer(data, dtype="<f4").copy() if data else np.zeros(0, np.float32)
                try:
                    greedy, kenlm, status, kenlm_label, accepted = engine.stream_pcm(
                        sid, samples, sr, offset, lm_mode=lm_mode, final=final
                    )
                except ValueError as exc:
                    _json(self, 400, {"error": str(exc)})
                    return
            else:
                greedy, kenlm, status, kenlm_label = engine.transcribe_bytes(
                    data, lm_mode=lm_mode, run_lm=True
                )
            too_short = status.startswith("Clip is")
            code = 400 if status.startswith("Could not") or (path.path != "/stream" and too_short) else 200
            ran_lm = kenlm_label.startswith("CTC +")
            payload = {
                "greedy": greedy,
                "kenlm": kenlm,
                "status": status,
                "has_lm": ran_lm or engine.kind == "zipformer" or engine.kenlm is not None,
                "lm_note": engine.lm_note,
                "greedy_label": engine.primary_label,
                "kenlm_label": kenlm_label,
            }
            if accepted is not None:
                payload["samples"] = accepted
                payload["pending"] = status.endswith("listening")
            _json(self, code, payload)

    return Handler


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=list(PRESETS), default=None)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--lm-alpha", type=float, default=0.5)
    p.add_argument("--lm-beta", type=float, default=0.8)
    p.add_argument("--host", default="0.0.0.0", help="Bind address (0.0.0.0 = LAN + localhost)")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument(
        "--https-port",
        type=int,
        default=0,
        help="Optional extra HTTPS-only port (0 = TLS on --port via HTTP/HTTPS dual stack)",
    )
    args = p.parse_args()

    cache: dict[str, LiveAsr] = {}
    specs: dict[str, dict] = {}
    if args.config and args.ckpt:
        specs["custom"] = {"label": str(args.ckpt), "config": Path(args.config), "ckpt": Path(args.ckpt)}
        default = "custom"
    else:
        wanted = _available_presets()
        if args.preset:
            if args.preset not in wanted:
                log.error("missing checkpoint for preset %s (%s)", args.preset, PRESETS[args.preset]["ckpt"])
                return 1
            wanted = {args.preset: wanted[args.preset]}
        if not wanted:
            log.error("no checkpoints found under exp/. Train first or pass --config and --ckpt.")
            return 1
        specs = wanted
        default = next(
            (
                k
                for k in (
                    "zipformer_s_pool_leyu",
                    "zipformer_s_pool_leyu_onnx",
                    "zipformer_s_pool_leyu_onnx_int8",
                    "zipformer_s_pool",
                    "zipformer_s_align",
                    "char12_pool_short",
                    "char12_pool",
                    "char12_pool_onnx",
                    "char12_pool_onnx_int8",
                    "char12",
                    "char12_onnx",
                    "char12_onnx_int8",
                    "char",
                    "bpe",
                )
                if k in specs
            ),
            next(iter(specs)),
        )

    def get_engine(key: str) -> LiveAsr:
        key = key if key in specs else default
        if key not in cache:
            spec = specs[key]
            cache[key] = LiveAsr(
                spec["config"],
                spec["ckpt"],
                lm_alpha=args.lm_alpha,
                lm_beta=args.lm_beta,
                lm_dirs=spec.get("lm_dirs"),
                char_lm_dir=spec.get("char_lm_dir"),
                onnx_paths=spec.get("onnx_paths"),
                onnx_ctc=spec.get("onnx_ctc"),
            )
        return cache[key]

    get_engine(default)
    lan = _lan_ipv4() if args.host in ("0.0.0.0", "::") else (None if args.host in ("127.0.0.1", "localhost") else args.host)
    ssl_ctx = None
    https_url = None
    extra_https = None
    try:
        cert, key = _ensure_tls(lan)
        ssl_ctx = _ssl_context(cert, key)
        if lan:
            https_url = f"https://{lan}:{args.port}"
    except (FileNotFoundError, subprocess.CalledProcessError, OSError) as exc:
        log.warning("TLS cert unavailable (%s). Phone mic needs HTTPS.", exc)

    if ssl_ctx and args.https_port and args.https_port != args.port:
        extra_https = ThreadingHTTPServer(
            (args.host, args.https_port),
            make_handler(get_engine, specs, default, https_url=https_url),
        )
        extra_https.socket = ssl_ctx.wrap_socket(extra_https.socket, server_side=True)
        threading.Thread(target=extra_https.serve_forever, daemon=True, name="demo-https").start()
        log.info("Extra HTTPS-only port: https://%s:%d", lan or "127.0.0.1", args.https_port)

    httpd = DualStackHTTPServer(
        (args.host, args.port),
        make_handler(
            get_engine,
            specs,
            default,
            https_url=https_url,
            redirect_lan_to_https=https_url is not None,
        ),
        ssl_context=ssl_ctx,
    )
    log.info("This machine:  http://127.0.0.1:%d", args.port)
    if https_url:
        log.info("Phone mic (same Wi-Fi):  %s", https_url)
        log.info("Accept the certificate warning once (Android: Advanced → Proceed).")
    elif lan:
        log.info("Phone (upload only):  http://%s:%d", lan, args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("stopped")
        if extra_https is not None:
            extra_https.shutdown()
    return 0
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
