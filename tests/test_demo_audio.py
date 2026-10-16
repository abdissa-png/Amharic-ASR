import numpy as np
import torch

from demo.app import _pad_edges, _prep_waveform, _to_mono16k
from models.features import SpecAugment
from models.features import SAMPLE_RATE


def test_to_mono16k_int16_stereo_and_resample():
    sr = 8000
    t = np.linspace(0, 1, sr, dtype=np.float64)
    stereo = (np.stack([np.sin(2 * np.pi * 220 * t), np.sin(2 * np.pi * 220 * t)], axis=1) * 16000).astype(np.int16)
    wav = _to_mono16k(sr, stereo)
    assert wav.ndim == 1
    assert wav.numel() == SAMPLE_RATE
    assert float(wav.abs().max()) <= 1.0 + 1e-5
    assert wav.dtype == torch.float32


def test_prep_waveform_dc_and_level():
    sr = SAMPLE_RATE
    t = torch.linspace(0, 1, sr)
    quiet = 0.005 * torch.sin(2 * torch.pi * 220 * t) + 0.2
    out = _prep_waveform(quiet, sr)
    assert float(out.mean().abs()) < 0.02
    assert 0.02 < float(out.pow(2).mean().sqrt()) < 0.2
    assert float(out.abs().max()) <= 1.0 + 1e-5


def test_pad_edges_adds_room_noise_not_digital_silence():
    sr = SAMPLE_RATE
    torch.manual_seed(0)
    wav = 0.002 * torch.randn(sr)
    wav[sr // 4 : 3 * sr // 4] += 0.3 * torch.sin(2 * torch.pi * 220 * torch.arange(sr // 2) / sr)
    out = _pad_edges(wav, sr, lead_s=0.3, tail_s=0.5)
    lead, tail = int(0.3 * sr), int(0.5 * sr)
    assert out.numel() == wav.numel() + lead + tail
    assert torch.equal(out[lead : lead + wav.numel()], wav)
    tail_rms = float(out[-tail:].pow(2).mean().sqrt())
    assert 0.0005 < tail_rms < 0.01


def test_specaug_time_mask_stays_inside_each_length():
    feats = torch.ones(2, 100, 80)
    feats[0, 20:] = 2
    lengths = torch.tensor([20, 100])
    aug = SpecAugment(freq_masks=0)
    for _ in range(20):
        out = aug(feats.clone(), lengths)
        assert torch.equal(out[0, 20:], feats[0, 20:])
        assert torch.all((out[0, :20] == 0) | (out[0, :20] == 1))


def test_prep_waveform_skips_gain_on_healthy_level():
    sr = SAMPLE_RATE
    t = torch.linspace(0, 1, sr)
    healthy = 0.1 * torch.sin(2 * torch.pi * 220 * t)
    out = _prep_waveform(healthy, sr)
    ratio = float(out.pow(2).mean().sqrt() / healthy.pow(2).mean().sqrt())
    assert 0.5 < ratio < 1.5
