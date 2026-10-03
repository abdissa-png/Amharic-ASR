import random

import torch

from data.augment import (
    AugmentConfig,
    WaveformAugment,
    apply_condition,
    apply_gain_db,
    convolve_rir,
    mix_noise,
    phone_codec,
    rms_db,
    set_rms_db,
    synthetic_rir,
)
from evaluation.robustness import group_by_field


def _tone(n=16000, freq=220.0, sr=16000):
    t = torch.arange(n, dtype=torch.float32) / sr
    return 0.2 * torch.sin(2 * torch.pi * freq * t)


def test_set_rms_db_hits_target():
    wav = set_rms_db(_tone(), -25.0)
    assert abs(rms_db(wav) - (-25.0)) < 0.3


def test_mix_noise_raises_energy():
    rng = random.Random(0)
    speech = _tone()
    noise = torch.randn_like(speech)
    mixed = mix_noise(speech, noise, snr_db=5.0, rng=rng)
    assert mixed.pow(2).mean() > speech.pow(2).mean()


def test_rir_changes_waveform_keeps_length():
    rng = random.Random(1)
    wav = _tone(8000)
    out = convolve_rir(wav, synthetic_rir(rng, rt60_s=0.3))
    assert out.numel() == wav.numel()
    assert not torch.allclose(out, wav)


def test_phone_codec_keeps_length():
    wav = _tone(16000)
    out = phone_codec(wav)
    assert out.numel() == wav.numel()


def test_apply_condition_clean_is_noop():
    wav = _tone()
    assert torch.equal(apply_condition(wav, "clean"), wav.float().reshape(-1))


def test_apply_condition_noisy_is_deterministic():
    wav = _tone()
    a = apply_condition(wav, "noisy", seed="clip_1")
    b = apply_condition(wav, "noisy", seed="clip_1")
    c = apply_condition(wav, "noisy", seed="clip_2")
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_augment_from_cfg_none_when_absent():
    assert AugmentConfig.from_train_cfg({}) is None
    assert AugmentConfig.from_train_cfg({"speed_perturb": [1.0]}) is None


def test_waveform_augment_synthetic_fallback():
    cfg = AugmentConfig(
        gain_db=(-30.0, -20.0),
        noise_prob=1.0,
        rir_prob=1.0,
        codec_prob=0.0,
        noise_dir=None,
        rir_dir=None,
        synthetic_if_empty=True,
    )
    aug = WaveformAugment(cfg)
    out = aug(_tone(), random.Random(0))
    assert out.numel() == 16000
    assert abs(rms_db(out) + 25.0) < 8.0  # uniform in [-30, -20]


def test_apply_gain_db_is_relative_and_keeps_peak():
    quiet = 0.01 * _tone()
    assert abs(rms_db(apply_gain_db(quiet, -6.0)) - (rms_db(quiet) - 6.0)) < 0.1
    assert abs(rms_db(apply_gain_db(quiet, 6.0)) - (rms_db(quiet) + 6.0)) < 0.1
    assert float(apply_gain_db(_tone(), 40.0).abs().max()) <= 0.99 + 1e-6


def test_relative_gain_follows_the_clip_level():
    cfg = AugmentConfig.from_train_cfg(
        {"augment": {"gain_db": False, "gain_rel_db": [-12, 6], "gain_rel_prob": 1.0,
                     "noise_prob": 0.0, "rir_prob": 0.0, "codec_prob": 0.0}}
    )
    assert cfg.gain_db is None and cfg.gain_rel_db == (-12.0, 6.0)
    aug = WaveformAugment(cfg)
    for level in (0.003, 0.2):
        wav = level * _tone()
        for seed in range(10):
            delta = rms_db(aug(wav, random.Random(seed))) - rms_db(wav)
            assert -12.1 <= delta <= 6.1


def test_speed_perturb_changes_duration():
    from models.features import speed_perturb

    wav = _tone(48000)
    assert speed_perturb(wav, 1.0).numel() == 48000
    assert abs(speed_perturb(wav, 0.9).numel() - 48000 / 0.9) <= 2
    assert abs(speed_perturb(wav, 1.1).numel() - 48000 / 1.1) <= 2


def test_train_drops_long_clips_eval_keeps_them(tmp_path):
    from data.dataset import AsrDataset
    from utils import write_jsonl

    path = tmp_path / "train.jsonl"
    write_jsonl(path, [{"clip_id": "a", "duration_s": 5.0}, {"clip_id": "b", "duration_s": 55.0}])
    train = AsrDataset(path, None, train=True, max_duration=40.0)
    assert [r["clip_id"] for r in train.rows] == ["a"] and train.dropped_long == 1
    dev = AsrDataset(path, None, train=False, max_duration=40.0)
    assert len(dev.rows) == 2 and dev.dropped_long == 0


def test_word_times_splits_on_the_space_token():
    import torch

    from data.screen_pairs import word_times

    seq = [0, 0, 3, 3, 0, 4, 0, 1, 0, 5, 5, 0, 6, 0, 0]
    lp = torch.full((len(seq), 8), -30.0)
    for t, tok in enumerate(seq):
        lp[t, tok] = 0.0
    targets = torch.tensor([3, 4, 1, 5, 6], dtype=torch.int32)
    assert word_times(lp, targets, space_id=1, frame_s=0.04, n_words=2) == [
        [0.08, 0.24],
        [0.36, 0.52],
    ]
    assert word_times(lp, targets, space_id=1, frame_s=0.04, n_words=3) is None


def test_crop_words_stays_inside_the_gaps():
    import random

    from data.dataset import SAMPLE_RATE, crop_words

    words = ["aa", "bb", "cc", "dd", "ee"]
    times = [[1.0, 1.4], [2.0, 2.5], [3.2, 3.6], [4.4, 4.9], [6.0, 6.5]]
    wav = torch.arange(int(8.0 * SAMPLE_RATE))
    cfg = {"prob": 1.0, "min_words": 1, "max_words": 3, "tight_tail_prob": 0.5}
    lengths = set()
    for n in range(40):
        out, text = crop_words(wav, " ".join(words), times, random.Random(n), cfg)
        got = text.split()
        i = next(j for j in range(len(words)) if words[j : j + len(got)] == got)
        start_s = float(out[0]) / SAMPLE_RATE
        end_s = float(out[-1]) / SAMPLE_RATE
        if i:
            assert start_s >= times[i - 1][1] - 1e-3
        if i + len(got) < len(words):
            assert end_s <= times[i + len(got)][0] + 1e-3
        assert end_s - start_s >= 0.34
        lengths.add(len(got))
    assert lengths == {1, 2, 3}
    same, text = crop_words(wav, " ".join(words), times, random.Random(0), {"prob": 0})
    assert text == " ".join(words) and same.numel() == wav.numel()
    _, text = crop_words(wav, " ".join(words), [[0.0, 1.0]], random.Random(0), cfg)
    assert text == " ".join(words)


def test_dataset_crops_in_training_only(tmp_path):
    import numpy as np
    import soundfile as sf

    from data.dataset import AsrDataset
    from utils import write_jsonl

    sf.write(tmp_path / "a.wav", np.zeros(16000 * 8, np.float32), 16000)
    text = "aa bb cc dd ee"
    write_jsonl(
        tmp_path / "train.jsonl",
        [
            {
                "clip_id": "c",
                "audio": "a.wav",
                "duration_s": 8.0,
                "text": text,
                "word_times": [[1.0, 1.4], [2.0, 2.5], [3.2, 3.6], [4.4, 4.9], [6.0, 6.5]],
            }
        ],
    )

    class Tok:
        def encode(self, s):
            return [1] * len(s.split())

    cfg = {"prob": 1.0, "min_words": 1, "max_words": 2, "tight_tail_prob": 0.0}
    train = AsrDataset(
        tmp_path / "train.jsonl",
        Tok(),
        train=True,
        root=tmp_path,
        max_duration=40,
        speed_perturb_factors=(1.0,),
        short_crop=cfg,
    )
    assert train.crop_ready == 1
    item = train[0]
    assert len(item["text"].split()) <= 2
    assert item["target_len"] == len(item["text"].split())
    dev = AsrDataset(
        tmp_path / "train.jsonl", Tok(), train=False, root=tmp_path, max_duration=40, short_crop=cfg
    )
    assert dev.short_crop is None and dev[0]["text"] == text


def test_group_by_region():
    rows = [
        {"clip_id": "a", "region": "addis"},
        {"clip_id": "b", "region": "sidama"},
        {"clip_id": "c", "region": ""},
    ]
    g = group_by_field(rows, "region")
    assert set(g) == {"addis", "sidama", "(not stated)"}
    assert len(g["addis"]) == 1
