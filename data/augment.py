"""Train-time waveform augmentation: gain, noise, reverb, phone codec.

SpecAugment stays in ``models.features`` (it is a Mel-domain mask). This
module runs on the 16 kHz waveform *before* ``log_mel``, which is the only
place level / SNR / reverb / codec mismatch can be repaired.

Noise and RIR files are optional. Empty dirs fall back to synthetic pink
noise and exponential-decay impulse responses so training does not depend
on a MUSAN download. Drop real files into ``data/noise`` / ``data/rir``
(or point the config at MUSAN) when you have them.

    python -m data.prepare_aug_assets
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from models.features import SAMPLE_RATE
from utils import ROOT

PEAK_CEIL = 0.99
_AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".opus", ".npy"}


def rms_db(wav: torch.Tensor) -> float:
    x = wav.detach().float().reshape(-1)
    return float(20.0 * torch.log10(x.pow(2).mean().sqrt().clamp_min(1e-12)))


def set_rms_db(wav: torch.Tensor, target_db: float, peak: float = PEAK_CEIL) -> torch.Tensor:
    x = wav.float()
    scale = 10.0 ** ((target_db - rms_db(x)) / 20.0)
    y = x * scale
    mx = float(y.abs().max()) if y.numel() else 0.0
    if mx > peak:
        y = y * (peak / mx)
    return y


def apply_gain_db(wav: torch.Tensor, gain_db: float, peak: float = PEAK_CEIL) -> torch.Tensor:
    """Scale by ``gain_db`` relative to the clip's own level."""
    y = wav.float() * (10.0 ** (gain_db / 20.0))
    mx = float(y.abs().max()) if y.numel() else 0.0
    if mx > peak:
        y = y * (peak / mx)
    return y


def _tile_to(wav: torch.Tensor, n: int, rng: random.Random) -> torch.Tensor:
    x = wav.reshape(-1)
    if x.numel() == 0:
        return torch.zeros(n, dtype=torch.float32)
    if x.numel() >= n:
        start = rng.randint(0, x.numel() - n) if x.numel() > n else 0
        return x[start : start + n].contiguous()
    reps = int(math.ceil(n / x.numel()))
    return x.repeat(reps)[:n].contiguous()


def mix_noise(speech: torch.Tensor, noise: torch.Tensor, snr_db: float, rng: random.Random) -> torch.Tensor:
    """Add noise at a target SNR (dB), keeping speech RMS."""
    s = speech.float().reshape(-1)
    n = _tile_to(noise.float(), s.numel(), rng)
    s_rms = s.pow(2).mean().sqrt().clamp_min(1e-12)
    n_rms = n.pow(2).mean().sqrt().clamp_min(1e-12)
    n = n * (s_rms / n_rms) * (10.0 ** (-float(snr_db) / 20.0))
    y = s + n
    mx = float(y.abs().max()) if y.numel() else 0.0
    if mx > PEAK_CEIL:
        y = y * (PEAK_CEIL / mx)
    return y


def convolve_rir(wav: torch.Tensor, rir: torch.Tensor) -> torch.Tensor:
    """FFT convolution; output length matches the input utterance."""
    x = wav.float().reshape(-1)
    h = rir.float().reshape(-1)
    if h.numel() < 2 or x.numel() == 0:
        return x
    h = h / h.pow(2).sum().sqrt().clamp_min(1e-12)
    n = x.numel() + h.numel() - 1
    nfft = 1 << (n - 1).bit_length()
    X = torch.fft.rfft(x, n=nfft)
    H = torch.fft.rfft(h, n=nfft)
    y = torch.fft.irfft(X * H, n=nfft)[: x.numel()]
    mx = float(y.abs().max()) if y.numel() else 0.0
    if mx > PEAK_CEIL:
        y = y * (PEAK_CEIL / mx)
    return y


def _gauss(n: int, rng: random.Random) -> torch.Tensor:
    """``n`` standard normal samples, seeded from ``rng`` so a seeded call repeats."""
    gen = torch.Generator().manual_seed(rng.getrandbits(63))
    return torch.randn(n, generator=gen, dtype=torch.float32)


def colored_noise(n: int, rng: random.Random, *, beta: float = 1.0) -> torch.Tensor:
    """Gaussian noise with power spectrum 1/f^beta (0=white, 1=pink)."""
    n = max(1, int(n))
    # Clip lengths after speed perturbation often have large prime factors,
    # which makes the FFT slow. A crop of longer noise has the same spectrum.
    n_fft = 1 << (n - 1).bit_length()
    real = _gauss(n_fft, rng)
    spec = torch.fft.rfft(real)
    freqs = torch.fft.rfftfreq(n_fft)
    mag = freqs.clamp_min(1.0 / n_fft).pow(-0.5 * beta)
    spec = spec * mag
    spec[0] = 0
    y = torch.fft.irfft(spec, n=n_fft)[:n]
    std = y.std().clamp_min(1e-6)
    return (y / std).contiguous()


def synthetic_rir(
    rng: random.Random,
    sample_rate: int = SAMPLE_RATE,
    rt60_s: float | None = None,
) -> torch.Tensor:
    """Exponentially decaying noise: a cheap room impulse response."""
    rt60 = float(rt60_s if rt60_s is not None else rng.uniform(0.15, 0.55))
    n = max(32, int(sample_rate * min(1.2, rt60 * 1.4)))
    noise = _gauss(n, rng)
    t = torch.arange(n, dtype=torch.float32) / float(sample_rate)
    # Intensity ~ exp(-t / tau); RT60 ≈ 6.91 tau for amplitude decay.
    tau = rt60 / 6.91
    env = torch.exp(-t / max(tau, 1e-4))
    h = noise * env
    h[0] = 1.0
    return h / h.pow(2).sum().sqrt().clamp_min(1e-12)


def phone_codec(wav: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Narrowband phone simulation: 8 kHz roundtrip + µ-law quantization.

    Fast enough for the training loop. Prefer this over ffmpeg Opus unless
    you are scoring a dedicated ``codec`` robustness slice.
    """
    import torchaudio

    x = wav.float().reshape(-1)
    if x.numel() < sample_rate // 50:
        return x
    y = torchaudio.functional.resample(x.unsqueeze(0), sample_rate, 8000)
    y = torchaudio.functional.mu_law_encoding(y.clamp(-1, 1), quantization_channels=256).float()
    y = torchaudio.functional.mu_law_decoding(y.long(), quantization_channels=256)
    y = torchaudio.functional.resample(y, 8000, sample_rate).squeeze(0)
    if y.numel() >= x.numel():
        return y[: x.numel()].contiguous()
    return torch.nn.functional.pad(y, (0, x.numel() - y.numel()))


def opus_codec(wav: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    from evaluation.match_et import opus_roundtrip

    y = opus_roundtrip(wav.detach().cpu().float().numpy(), sample_rate=sample_rate)
    out = torch.from_numpy(y)
    n = wav.numel()
    if out.numel() >= n:
        return out[:n]
    return torch.nn.functional.pad(out, (0, n - out.numel()))


def list_audio_files(root: Path | None) -> list[Path]:
    if root is None:
        return []
    path = Path(root)
    if not path.is_dir():
        return []
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in _AUDIO_SUFFIXES)


def load_wav_file(path: Path, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    import numpy as np
    import soundfile as sf

    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    wav = torch.from_numpy(np.ascontiguousarray(arr))
    if wav.dim() == 2:
        wav = wav.mean(dim=-1)
    if int(sr) != sample_rate:
        import torchaudio

        wav = torchaudio.functional.resample(wav.unsqueeze(0), int(sr), sample_rate).squeeze(0)
    return wav.float()


@dataclass
class AugmentConfig:
    gain_db: tuple[float, float] | None = (-35.0, -15.0)
    # Random gain around the clip's own RMS, Kaldi volume-perturbation style.
    # Unlike gain_db it keeps each corpus near the level CMVN was fit to.
    gain_rel_db: tuple[float, float] | None = None
    gain_rel_prob: float = 0.5
    noise_prob: float = 0.5
    noise_snr_db: tuple[float, float] = (5.0, 20.0)
    noise_dir: Path | None = None
    rir_prob: float = 0.3
    rir_dir: Path | None = None
    codec_prob: float = 0.2
    codec: str = "phone"
    synthetic_if_empty: bool = True

    @classmethod
    def from_train_cfg(cls, tcfg: dict[str, Any] | None) -> "AugmentConfig | None":
        if not tcfg:
            return None
        raw = tcfg.get("augment")
        if not raw:
            return None
        if raw is True:
            raw = {}
        if not isinstance(raw, dict):
            raise TypeError("train.augment must be a mapping or true")
        gain = raw.get("gain_db", [-35.0, -15.0])
        gain_rel = raw.get("gain_rel_db")
        snr = raw.get("noise_snr_db", [5.0, 20.0])
        noise_dir = raw.get("noise_dir", "data/noise")
        rir_dir = raw.get("rir_dir", "data/rir")
        return cls(
            gain_db=None if gain in (None, False) else (float(gain[0]), float(gain[1])),
            gain_rel_db=None if gain_rel in (None, False) else (float(gain_rel[0]), float(gain_rel[1])),
            gain_rel_prob=float(raw.get("gain_rel_prob", 0.5)),
            noise_prob=float(raw.get("noise_prob", 0.5)),
            noise_snr_db=(float(snr[0]), float(snr[1])),
            noise_dir=ROOT / noise_dir if noise_dir else None,
            rir_prob=float(raw.get("rir_prob", 0.3)),
            rir_dir=ROOT / rir_dir if rir_dir else None,
            codec_prob=float(raw.get("codec_prob", 0.2)),
            codec=str(raw.get("codec", "phone")),
            synthetic_if_empty=bool(raw.get("synthetic_if_empty", True)),
        )


class WaveformAugment:
    """Randomized multi-condition chain used only on the training split."""

    def __init__(self, cfg: AugmentConfig) -> None:
        self.cfg = cfg
        self.noise_files = list_audio_files(cfg.noise_dir)
        self.rir_files = list_audio_files(cfg.rir_dir)

    def _noise(self, n: int, rng: random.Random) -> torch.Tensor:
        if self.noise_files:
            return load_wav_file(rng.choice(self.noise_files))
        if self.cfg.synthetic_if_empty:
            return colored_noise(n, rng, beta=rng.choice([0.0, 1.0, 1.5]))
        return torch.zeros(n)

    def _rir(self, rng: random.Random) -> torch.Tensor:
        if self.rir_files:
            return load_wav_file(rng.choice(self.rir_files))
        return synthetic_rir(rng)

    def __call__(self, wav: torch.Tensor, rng: random.Random | None = None) -> torch.Tensor:
        rng = rng or random
        y = wav.float().reshape(-1)
        if y.numel() == 0:
            return y
        if rng.random() < self.cfg.rir_prob:
            y = convolve_rir(y, self._rir(rng))
        if rng.random() < self.cfg.noise_prob:
            snr = rng.uniform(*self.cfg.noise_snr_db)
            y = mix_noise(y, self._noise(y.numel(), rng), snr, rng)
        if self.cfg.gain_db is not None:
            y = set_rms_db(y, rng.uniform(*self.cfg.gain_db))
        if self.cfg.gain_rel_db is not None and rng.random() < self.cfg.gain_rel_prob:
            y = apply_gain_db(y, rng.uniform(*self.cfg.gain_rel_db))
        if rng.random() < self.cfg.codec_prob:
            if self.cfg.codec == "opus":
                try:
                    y = opus_codec(y)
                except Exception:
                    y = phone_codec(y)
            else:
                y = phone_codec(y)
        return y


def apply_condition(
    wav: torch.Tensor,
    condition: str,
    *,
    seed: int | str = 0,
    sample_rate: int = SAMPLE_RATE,
    noise_dir: Path | None = None,
    rir_dir: Path | None = None,
) -> torch.Tensor:
    """Deterministic eval perturbation. ``clean`` is a no-op."""
    y = wav.float().reshape(-1)
    if condition in ("clean", "clean_read", "identity", ""):
        return y
    rng = random.Random(str(seed) + "|" + condition)
    if condition in ("noisy", "noise"):
        files = list_audio_files(noise_dir)
        noise = load_wav_file(rng.choice(files)) if files else colored_noise(y.numel(), rng, beta=1.0)
        return mix_noise(y, noise, rng.uniform(5.0, 15.0), rng)
    if condition in ("reverb", "rir"):
        files = list_audio_files(rir_dir)
        rir = load_wav_file(rng.choice(files)) if files else synthetic_rir(rng, sample_rate, rt60_s=0.4)
        return convolve_rir(y, rir)
    if condition in ("gain", "gain_low"):
        return set_rms_db(y, -40.0)
    if condition == "gain_et":
        return set_rms_db(y, -25.0)
    if condition == "gain_high":
        return set_rms_db(y, -18.0)
    if condition == "gain_rand":
        return set_rms_db(y, rng.uniform(-35.0, -15.0))
    if condition in ("codec", "device_mismatch"):
        return phone_codec(y, sample_rate)
    if condition == "codec_opus":
        try:
            return opus_codec(y, sample_rate)
        except Exception:
            return phone_codec(y, sample_rate)
    if condition.startswith("speed_"):
        from models.features import speed_perturb

        factor = float(condition.split("_", 1)[1])
        return speed_perturb(y, factor, sample_rate)
    raise ValueError(f"unknown robustness condition {condition!r}")
