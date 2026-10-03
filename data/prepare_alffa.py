"""Download ALFFA Amharic read speech (~20 h train / ~2 h test).

Source: OpenSLR SLR25 / Hugging Face ``hadamard-2/alffa-amharic``.
This is the AMH-2005 read corpus — clean, speaker-overlapping, not Telegram
phone speech. Pool it with Dataset.ET; do not treat its WER as in-domain.

    python -m data.prepare_alffa

Writes::

    data/alffa_am/{train,dev,test}/*.flac
    metadata/alffa_am/{train,dev,test}.jsonl
    metadata/alffa_am/stats.json

The upstream release has train/test only. 5% of train is carved out as
``dev`` (seed 0) so early-stopping jsonl exists; official test is untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

from data.normalize import for_eval, for_training
from data.prepare import SAMPLE_RATE, write_audio_flac
from utils import ROOT, save_json, setup_logging, write_jsonl

HF_DATASET = "hadamard-2/alffa-amharic"
DEV_FRACTION = 0.05
DEV_SEED = 0

log = setup_logging()


def _record(utt: str, audio_rel: str, text: str, speaker: str) -> dict[str, Any]:
    return {
        "audio": audio_rel,
        "text": for_training(text),
        "text_raw": text,
        "text_eval": for_eval(text),
        "speaker_id": speaker or "",
        "prompt_id": f"alffa_{utt}",
        "clip_id": f"alffa_{utt}",
        "gender": "",
        "age_band": "",
        "region": "",
        "duration_s": 0.0,
        "speech_s": 0.0,
        "speech_start_s": 0.0,
        "speech_end_s": 0.0,
        "lufs": None,
        "sample_rate": SAMPLE_RATE,
        "up_votes": 0,
        "down_votes": 0,
        "language": "am",
        "corpus": "alffa",
    }


def _duration(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def _iter_split(split: str, cache_dir: str | None):
    from datasets import Audio, load_dataset

    ds = load_dataset(HF_DATASET, split=split, cache_dir=cache_dir)
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE, decode=False))
    for row in ds:
        yield dict(row)


def _carve_dev(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import random

    rng = random.Random(DEV_SEED)
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    n_dev = max(1, int(len(rows) * DEV_FRACTION))
    dev_i = set(idxs[:n_dev])
    train, dev = [], []
    for i, row in enumerate(rows):
        (dev if i in dev_i else train).append(row)
    return train, dev


def prepare(
    *,
    out_dir: Path = ROOT,
    cache_dir: str | None = None,
    skip_existing: bool = True,
) -> dict[str, Any]:
    meta_root = out_dir / "metadata" / "alffa_am"
    meta_root.mkdir(parents=True, exist_ok=True)
    raw: dict[str, list[dict[str, Any]]] = {}
    for hf_split in ("train", "test"):
        records: list[dict[str, Any]] = []
        log.info("loading %s", hf_split)
        for i, row in enumerate(_iter_split(hf_split, cache_dir)):
            utt = str(row.get("utterance_id") or row.get("id") or f"{hf_split}_{i:05d}")
            text = row.get("transcript") or row.get("transcription") or row.get("text") or ""
            speaker = str(row.get("speaker_id") or "")
            local = "test" if hf_split == "test" else "train"
            rel = f"data/alffa_am/{local}/{utt}.flac"
            dest = out_dir / rel
            rec = _record(utt, rel, str(text), speaker)
            if not (skip_existing and dest.exists()):
                rec["duration_s"] = write_audio_flac(dest, row.get("audio"))
            else:
                rec["duration_s"] = _duration(dest)
            rec["speech_s"] = rec["duration_s"]
            rec["speech_end_s"] = rec["duration_s"]
            records.append(rec)
            if (i + 1) % 500 == 0:
                log.info("  %s %d", hf_split, i + 1)
        records.sort(key=lambda r: r["clip_id"])
        raw[hf_split] = records
        log.info("%s %d clips, %.2f h", hf_split, len(records), sum(r["duration_s"] for r in records) / 3600.0)

    train, dev = _carve_dev(raw["train"])
    # Point carved-dev audio at the train folder files they already occupy.
    all_rows = {"train": train, "dev": dev, "test": raw["test"]}
    for name, rows in all_rows.items():
        write_jsonl(meta_root / f"{name}.jsonl", rows)
    stats = {
        "dataset": HF_DATASET,
        "note": "dev is 5% of upstream train (seed 0); test is official ALFFA test",
        "splits": {},
    }
    for name, rows in all_rows.items():
        stats["splits"][name] = {
            "clips": len(rows),
            "hours": round(sum(r["duration_s"] for r in rows) / 3600.0, 3),
            "speakers": len({r["speaker_id"] for r in rows}),
        }
    save_json(meta_root / "stats.json", stats)
    return stats


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=ROOT)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--no-skip-existing", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)
    prepare(out_dir=args.out_dir, cache_dir=args.cache_dir, skip_existing=not args.no_skip_existing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
