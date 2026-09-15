"""Log-mel features, SpecAugment, speed perturbation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio


SAMPLE_RATE = 16000
N_MELS = 80
FRAME_MS = 25.0
HOP_MS = 10.0


def speed_perturb(waveform: torch.Tensor, factor: float, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    if abs(factor - 1.0) < 1e-6:
        return waveform
    # Treat the samples as recorded at sample_rate * factor and resample once:
    # the clip plays at ``factor``× speed (length / factor). A round trip
    # through the new rate would only low-pass it and keep the length.
    return torchaudio.functional.resample(waveform, int(round(sample_rate * factor)), sample_rate)


def log_mel(
    waveform: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    dither: float = 1e-5,
) -> torch.Tensor:
    """(T,) or (1, T) waveform → (frames, n_mels) log-mel."""
    if waveform.dim() == 2:
        waveform = waveform.mean(0)
    wav = waveform.unsqueeze(0) if waveform.dim() == 1 else waveform
    feats = torchaudio.compliance.kaldi.fbank(
        wav,
        num_mel_bins=n_mels,
        frame_length=FRAME_MS,
        frame_shift=HOP_MS,
        sample_frequency=sample_rate,
        dither=dither,
        energy_floor=1.0,
        use_energy=False,
    )
    return feats  # (frames, n_mels)


class GlobalCMVN:
    """Global mean/variance on log-Mel, ESPNet-style."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = mean.float().view(1, -1)
        self.std = std.float().view(1, -1).clamp_min(1e-5)

    def __call__(self, feats: torch.Tensor) -> torch.Tensor:
        return (feats - self.mean.to(feats.device, feats.dtype)) / self.std.to(feats.device, feats.dtype)

    def save(self, path) -> None:
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mean": self.mean.squeeze(0).cpu(), "std": self.std.squeeze(0).cpu()}, path)

    @classmethod
    def load(cls, path) -> "GlobalCMVN":
        blob = torch.load(path, map_location="cpu", weights_only=True)
        return cls(blob["mean"], blob["std"])


def compute_cmvn(jsonl, *, root, n_utts: int = 1500, seed: int = 0) -> GlobalCMVN:
    """Estimate CMVN from a random subset of training clips."""
    import random

    from utils import load_jsonl

    rows = load_jsonl(jsonl)
    rng = random.Random(seed)
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    total = torch.zeros(N_MELS)
    total_sq = torch.zeros(N_MELS)
    count = 0
    for i in idxs[: min(n_utts, len(idxs))]:
        import numpy as np
        import soundfile as sf

        path = root / rows[i]["audio"]
        wav_np, sr = sf.read(str(path), dtype="float32", always_2d=False)
        wav = torch.from_numpy(np.ascontiguousarray(wav_np))
        if wav.dim() == 2:
            wav = wav.mean(dim=-1)
        if int(sr) != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), int(sr), SAMPLE_RATE).squeeze(0)
        feats = log_mel(wav)
        total += feats.sum(0)
        total_sq += (feats * feats).sum(0)
        count += feats.size(0)
    mean = total / max(1, count)
    var = total_sq / max(1, count) - mean * mean
    return GlobalCMVN(mean, var.clamp_min(0).sqrt())


class SpecAugment(nn.Module):
    def __init__(self, freq_masks: int = 2, time_masks: int = 2, freq_width: int = 27, time_width: int = 40) -> None:
        super().__init__()
        self.freq_masks = freq_masks
        self.time_masks = time_masks
        self.freq_width = freq_width
        self.time_width = time_width

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        # feats: (B, T, F) or (T, F). Time masks stay inside ``lengths`` so a
        # short clip in a padded batch is not wiped by a mask sized to the pad.
        squeeze = False
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
            squeeze = True
        b, t, f = feats.shape
        out = feats.clone()
        for _ in range(self.freq_masks):
            w = int(torch.randint(0, self.freq_width + 1, (1,)).item())
            if w == 0 or f <= w:
                continue
            f0 = int(torch.randint(0, f - w + 1, (1,)).item())
            out[:, :, f0 : f0 + w] = 0
        if lengths is None:
            speech = [t] * b
        else:
            speech = [int(n) for n in lengths.detach().cpu().tolist()]
        for i, ti in enumerate(speech):
            ti = max(0, min(int(ti), t))
            if ti <= 1:
                continue
            max_tw = min(self.time_width, max(1, ti // 5))
            for _ in range(self.time_masks):
                w = int(torch.randint(0, max_tw + 1, (1,)).item())
                if w == 0 or ti <= w:
                    continue
                t0 = int(torch.randint(0, ti - w + 1, (1,)).item())
                out[i, t0 : t0 + w, :] = 0
        return out.squeeze(0) if squeeze else out
