"""PyTorch Dataset over frozen metadata jsonl + flac files."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from models.features import SAMPLE_RATE, log_mel, speed_perturb
from utils import ROOT, load_jsonl


def crop_words(
    wav: torch.Tensor,
    text: str,
    word_times: list | None,
    rng: random.Random,
    cfg: dict,
) -> tuple[torch.Tensor, str]:
    """Replace a clip with 1–3 of its words, cut inside the gaps around them.

    ``word_times`` is ``[start_s, end_s]`` per word from a CTC alignment. The
    cut starts after the previous word and ends before the next one, so the
    transcript of the crop is exactly the audio it contains. A tight tail
    (``tight_tail_prob``) ends soon after the last word, which is the case a
    live recording produces when the user stops on the last syllable.
    """
    if rng.random() >= float(cfg.get("prob", 0.0)):
        return wav, text
    words = text.split()
    dur = wav.numel() / SAMPLE_RATE
    hi = min(int(cfg.get("max_words", 3)), len(words) - 1)
    lo = int(cfg.get("min_words", 1))
    if not _times_ok(word_times, len(words), dur) or hi < lo:
        return wav, text
    tight_p = float(cfg.get("tight_tail_prob", 0.35))
    for _ in range(4):
        k = rng.randint(lo, hi)
        i = rng.randrange(0, len(words) - k + 1)
        bounds = _crop_bounds(word_times, i, k, dur, rng, tight_p)
        if bounds is None:
            continue
        start, end = bounds
        s0 = max(0, min(wav.numel() - 1, int(round(start * SAMPLE_RATE))))
        s1 = max(s0 + 1, min(wav.numel(), int(round(end * SAMPLE_RATE))))
        return wav[s0:s1], " ".join(words[i : i + k])
    return wav, text


def _times_ok(times, n_words: int, dur: float) -> bool:
    if not times or len(times) != n_words:
        return False
    prev = 0.0
    for start, end in times:
        if not (0.0 <= float(start) < float(end) <= dur + 0.05) or float(start) < prev - 1e-3:
            return False
        prev = float(start)
    return True


def _crop_bounds(times, i: int, k: int, dur: float, rng: random.Random, tight_p: float):
    """Seconds ``[start, end]`` around words ``i .. i+k``, inside the gaps."""
    t0, t1 = float(times[i][0]), float(times[i + k - 1][1])
    prev_end = float(times[i - 1][1]) if i else 0.0
    next_start = float(times[i + k][0]) if i + k < len(times) else None
    lead_room = max(0.0, t0 - (prev_end + 0.02 if i else 0.0))
    tail_room = max(0.0, (dur if next_start is None else next_start) - t1)
    if next_start is not None and tail_room < 0.08:
        return None
    lead_hi = min(0.25, max(0.05, lead_room))
    lead = rng.uniform(0.05, lead_hi) if lead_hi > 0.05 else min(0.05, lead_room)
    tight = rng.random() < tight_p
    tail_lo, tail_cap = (0.10, 0.18) if tight else (0.15, 0.45)
    tail_hi = min(tail_cap, tail_room)
    if tail_hi < tail_lo:
        return None
    start = max(0.0, t0 - rng.uniform(0.0, lead) - 0.02)
    if i:
        start = max(start, prev_end + 0.02)
    end = min(dur, t1 + rng.uniform(tail_lo, tail_hi))
    if next_start is not None:
        end = min(end, next_start - 0.02)
    if end - start < 0.35:
        return None
    return start, end


class AsrDataset(Dataset):
    def __init__(
        self,
        jsonl: str | Path,
        tokenizer,
        *,
        root: Path | None = None,
        train: bool = False,
        speed_perturb_factors: tuple[float, ...] = (1.0,),
        trim_silence: bool = False,
        max_duration: float | None = 30.0,
        cmvn=None,
        max_utts: int | None = None,
        subset_seed: int = 0,
        match_et: str | None = None,
        augment=None,
        robustness: str | None = None,
        noise_dir: Path | None = None,
        rir_dir: Path | None = None,
        short_crop: dict | None = None,
        eval_speed: float = 1.0,
    ) -> None:
        self.rows = load_jsonl(jsonl)
        # Audio is never cut: a prefix scored or trained against the full
        # transcript is a wrong label. Training drops over-long clips instead;
        # evaluation keeps every clip whole.
        self.dropped_long = 0
        if train and max_duration:
            keep = [r for r in self.rows if float(r.get("duration_s") or 0.0) <= max_duration]
            self.dropped_long = len(self.rows) - len(keep)
            self.rows = keep
        if max_utts is not None and max_utts < len(self.rows):
            rng = random.Random(subset_seed)
            self.rows = rng.sample(self.rows, max_utts)
            self.rows.sort(key=lambda r: r.get("clip_id") or "")
        self.tokenizer = tokenizer
        self.root = root or ROOT
        self.train = train
        self.speed_perturb_factors = speed_perturb_factors
        self.trim_silence = trim_silence
        self.max_duration = max_duration
        self.cmvn = cmvn
        self.match_et = match_et
        self.augment = augment
        self.robustness = robustness
        self.noise_dir = noise_dir
        self.rir_dir = rir_dir
        self.short_crop = short_crop if train else None
        self.eval_speed = float(eval_speed)
        self.crop_ready = sum(1 for r in self.rows if r.get("word_times"))

    def __len__(self) -> int:
        return len(self.rows)

    def _load_wav(self, row: dict[str, Any]) -> torch.Tensor:
        import numpy as np
        import soundfile as sf

        path = self.root / row["audio"]
        wav_np, sr = sf.read(str(path), dtype="float32", always_2d=False)
        wav = torch.from_numpy(np.ascontiguousarray(wav_np))
        if wav.dim() == 2:
            wav = wav.mean(dim=-1)
        if int(sr) != SAMPLE_RATE:
            import torchaudio

            wav = torchaudio.functional.resample(wav.unsqueeze(0), int(sr), SAMPLE_RATE).squeeze(0)
        if self.match_et:
            from evaluation.match_et import apply_match

            wav = apply_match(wav, self.match_et, sample_rate=SAMPLE_RATE)
        if self.trim_silence:
            start = int(float(row.get("speech_start_s") or 0.0) * SAMPLE_RATE)
            end = int(float(row.get("speech_end_s") or 0.0) * SAMPLE_RATE)
            if end > start + SAMPLE_RATE // 4:
                pad = int(0.05 * SAMPLE_RATE)
                wav = wav[max(0, start - pad) : min(wav.numel(), end + pad)]
        if self.robustness:
            from data.augment import apply_condition

            wav = apply_condition(
                wav,
                self.robustness,
                seed=row.get("clip_id") or 0,
                sample_rate=SAMPLE_RATE,
                noise_dir=self.noise_dir,
                rir_dir=self.rir_dir,
            )
        if not self.train and abs(self.eval_speed - 1.0) > 1e-6:
            wav = speed_perturb(wav, self.eval_speed)
        return wav

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        wav = self._load_wav(row)
        text = row["text"]
        if self.short_crop and row.get("word_times"):
            wav, text = crop_words(wav, text, row["word_times"], random, self.short_crop)
        if self.train and self.speed_perturb_factors:
            factors = list(self.speed_perturb_factors)
            if self.max_duration:
                max_n = self.max_duration * SAMPLE_RATE
                factors = [f for f in factors if wav.numel() / f <= max_n] or [1.0]
            wav = speed_perturb(wav, random.choice(factors))
        if self.train and self.augment is not None:
            wav = self.augment(wav)
        feats = log_mel(wav)
        if self.cmvn is not None:
            feats = self.cmvn(feats)
        ids = self.tokenizer.encode(text)
        return {
            "feats": feats,
            "feat_len": feats.size(0),
            "targets": torch.tensor(ids, dtype=torch.long),
            "target_len": len(ids),
            "duration_s": float(row.get("duration_s") or wav.numel() / SAMPLE_RATE),
            "clip_id": row["clip_id"],
            "text": text,
            "text_raw": text if text != row["text"] else row.get("text_raw", text),
            "speaker_id": row.get("speaker_id", ""),
            "prompt_id": row.get("prompt_id", ""),
            "region": row.get("region", ""),
            "corpus": row.get("corpus", ""),
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    feats = pad_sequence([b["feats"] for b in batch], batch_first=True, padding_value=0.0)
    targets = pad_sequence([b["targets"] for b in batch], batch_first=True, padding_value=0)
    return {
        "feats": feats,
        "feat_lens": torch.tensor([b["feat_len"] for b in batch], dtype=torch.long),
        "targets": targets,
        "target_lens": torch.tensor([b["target_len"] for b in batch], dtype=torch.long),
        "duration_s": [b["duration_s"] for b in batch],
        "clip_id": [b["clip_id"] for b in batch],
        "text": [b["text"] for b in batch],
        "text_raw": [b["text_raw"] for b in batch],
        "speaker_id": [b["speaker_id"] for b in batch],
        "prompt_id": [b["prompt_id"] for b in batch],
        "region": [b["region"] for b in batch],
        "corpus": [b["corpus"] for b in batch],
    }


def pack_by_padded_seconds(durations: list[float], order: list[int], max_seconds: float) -> list[list[int]]:
    """Group indices so ``count × longest clip`` stays within ``max_seconds``.

    Collate pads every utterance to the longest one in the batch, so a sum
    of durations under the budget can still be a much larger GPU tensor.
    A clip longer than the budget is its own batch.
    """
    batches: list[list[int]] = []
    batch: list[int] = []
    peak = 0.0
    for i in order:
        d = max(0.5, float(durations[i]))
        if batch and max(peak, d) * (len(batch) + 1) > max_seconds:
            batches.append(batch)
            batch = []
            peak = 0.0
        batch.append(i)
        peak = max(peak, d)
    if batch:
        batches.append(batch)
    return batches


class DurationBatchSampler:
    """Pack utterances until the padded batch is ``max_seconds`` of audio."""

    def __init__(self, dataset: AsrDataset, max_seconds: float, shuffle: bool = True, seed: int = 0) -> None:
        self.dataset = dataset
        self.max_seconds = max_seconds
        self.shuffle = shuffle
        self.seed = seed
        self.durations = [float(r.get("duration_s") or 0.0) for r in dataset.rows]

    def __iter__(self):
        idxs = list(range(len(self.dataset)))
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(idxs)
            self.seed += 1
        yield from pack_by_padded_seconds(self.durations, idxs, self.max_seconds)

    def __len__(self) -> int:
        # Approximate; actual count depends on packing.
        total = sum(self.durations) or 1.0
        return max(1, int(total / self.max_seconds))
