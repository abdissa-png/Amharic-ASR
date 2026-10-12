"""Bring out-of-domain audio closer to Dataset.ET's Telegram/phone acoustics.

Dataset.ET is crowd-sourced Telegram Opus at 16 kHz, not loudness-normalised
(median LUFS ≈ -28, median RMS ≈ -25 dBFS). FLEURS am_et is the same sample
rate and duration, but ~18 dB quieter at the median and spectrally brighter
(phone-band vs studio). Global CMVN cannot recover speech that has already
hit Kaldi's ``energy_floor``.
"""

from __future__ import annotations

import subprocess
from typing import Literal

import numpy as np
import torch

MatchKind = Literal["gain", "gain_opus"]

# Median RMS of a Dataset.ET test shard (see the FLEURS comparison script).
TARGET_RMS_DB = -25.0
PEAK_CEIL = 0.99
OPUS_BITRATE = "24k"


def _to_numpy(wav: torch.Tensor) -> np.ndarray:
    x = wav.detach().cpu().float().numpy()
    if x.ndim > 1:
        x = x.mean(axis=0)
    return np.ascontiguousarray(x, dtype=np.float32)


def rms_db(x: np.ndarray) -> float:
    return float(20.0 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-12))


def gain_to_rms(x: np.ndarray, target_db: float = TARGET_RMS_DB) -> np.ndarray:
    """Peak-limited gain so RMS matches Dataset.ET's typical phone level."""
    current = rms_db(x)
    scale = 10.0 ** ((target_db - current) / 20.0)
    y = x * np.float32(scale)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > PEAK_CEIL:
        y *= np.float32(PEAK_CEIL / peak)
    return y


def opus_roundtrip(x: np.ndarray, sample_rate: int = 16000, bitrate: str = OPUS_BITRATE) -> np.ndarray:
    """Telegram-like Opus encode/decode (SILK-ish 16 kHz voice)."""
    enc = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            "f32le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "libopus",
            "-b:a",
            bitrate,
            "-application",
            "voip",
            "-f",
            "ogg",
            "pipe:1",
        ],
        input=x.astype(np.float32, copy=False).tobytes(),
        capture_output=True,
        check=True,
    )
    dec = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            "ogg",
            "-i",
            "pipe:0",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "pipe:1",
        ],
        input=enc.stdout,
        capture_output=True,
        check=True,
    )
    y = np.frombuffer(dec.stdout, dtype=np.float32)
    if y.size == 0:
        return x
    return y.copy()


def apply_match(wav: torch.Tensor, kind: MatchKind, sample_rate: int = 16000) -> torch.Tensor:
    x = _to_numpy(wav)
    if kind == "gain":
        y = gain_to_rms(x)
    elif kind == "gain_opus":
        y = gain_to_rms(x)
        y = opus_roundtrip(y, sample_rate=sample_rate)
        y = gain_to_rms(y)
    else:
        raise ValueError(f"unknown match kind {kind}")
    return torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
