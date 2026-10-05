"""Download the Leyu Amharic dialect corpora (mobile, paired, region-labeled).

Five Hugging Face datasets under ``leyu-amharic``. Cards claim roughly:

    addis-ababa  124 h     gonder  113 h     shewa  102 h
    gojjam        82 h     wello    19 h

The same prompt is read by many speakers, so a random utterance split leaks
text into dev. Splits are by normalised prompt: 5% of prompts → dev, 5% →
test, the rest → train (seed 0). Every speaker of a held-out prompt stays
out of train.

Clips run to ~80 s. The pool config raises ``max_duration`` so the loader
does not keep a full transcript against a 30 s audio prefix.

Audio is not kept in memory. A first pass records text only, then each
clip is decoded and written to FLAC on its own. Holding every dialect's
waveform until the end exhausted RAM before any file was written.

    python -m data.prepare_leyu
    python -m data.prepare_leyu --dialects addis-ababa gojjam

Writes::

    data/leyu_am/{train,dev,test}/*.flac
    metadata/leyu_am/{train,dev,test}.jsonl
    metadata/leyu_am/stats.json
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from data.normalize import for_eval, for_training
from data.prepare import SAMPLE_RATE, write_audio_flac
from utils import ROOT, save_json, setup_logging, write_jsonl

HF_DATASETS = {
    "addis-ababa": "leyu-amharic/leyu-amharic-addis-ababa-dialect",
    "gojjam": "leyu-amharic/leyu-amharic-gojjam-dialect",
    "wello": "leyu-amharic/leyu-amharic-wello-dialect",
    "gonder": "leyu-amharic/leyu-amharic-gonder-dialect",
    "shewa": "leyu-amharic/leyu-amharic-shewa-dialect",
}
DEV_FRACTION = 0.05
TEST_FRACTION = 0.05
SPLIT_SEED = 0

log = setup_logging()


def assign_prompts(
    prompts: Iterable[str],
    *,
    dev_frac: float = DEV_FRACTION,
    test_frac: float = TEST_FRACTION,
    seed: int = SPLIT_SEED,
) -> dict[str, str]:
    """Map each distinct prompt to train, dev, or test. No prompt is in two splits."""
    uniq = sorted({p for p in prompts if p})
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    if n == 0:
        return {}
    n_dev = int(round(n * dev_frac))
    n_test = int(round(n * test_frac))
    if n >= 20:
        n_dev = max(1, n_dev)
        n_test = max(1, n_test)
    if n_dev + n_test >= n:
        n_dev = max(0, n // 10)
        n_test = max(0, n // 10)
    out: dict[str, str] = {}
    for i, prompt in enumerate(uniq):
        if i < n_dev:
            out[prompt] = "dev"
        elif i < n_dev + n_test:
            out[prompt] = "test"
        else:
            out[prompt] = "train"
    return out


def _duration(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def _row_text(row: dict[str, Any]) -> str:
    return str(row.get("text") or row.get("transcription") or row.get("sentence") or "")


def _load_splits(repo: str, cache_dir: str | None):
    """Open each split. Audio stays as bytes; callers must not retain those bytes."""
    from datasets import Audio, get_dataset_split_names, load_dataset

    names = list(get_dataset_split_names(repo))
    # Leyu cards are a single split; still honour train/validation/test if present.
    # verification_mode: a stale HF cache can record an older example count
    # (Addis Ababa was 4,862, now 9,854) and abort after a good generate.
    for name in names:
        ds = load_dataset(
            repo,
            split=name,
            cache_dir=cache_dir,
            verification_mode="no_checks",
        )
        if "audio" in ds.column_names:
            ds = ds.cast_column("audio", Audio(decode=False))
        yield name, ds


def _collect_clips(dialect: str, repo: str, cache_dir: str | None) -> list[dict[str, Any]]:
    """Text and speaker fields only. The audio column is not copied into Python."""
    import gc

    kept: list[dict[str, Any]] = []
    for split_name, ds in _load_splits(repo, cache_dir):
        columns = [c for c in ds.column_names if c != "audio"]
        meta = ds.select_columns(columns)
        n = 0
        for i in range(len(meta)):
            row = meta[i]
            text = _row_text(row)
            train_text = for_training(text)
            if not train_text:
                continue
            kept.append(
                {
                    "text": train_text,
                    "text_raw": text,
                    "text_eval": for_eval(text),
                    "speaker_id": str(row.get("speaker_id") or ""),
                    "gender": str(row.get("gender") or ""),
                    "region": str(row.get("dialect") or dialect),
                    "dialect": dialect,
                    "hf_split": split_name,
                    "hf_index": i,
                }
            )
            n += 1
            if n % 500 == 0:
                log.info("  %s %d", dialect, n)
        del meta, ds
        gc.collect()
    log.info("%s kept %d non-empty clips", dialect, len(kept))
    return kept


def prepare(
    *,
    dialects: Iterable[str] | None = None,
    out_dir: Path = ROOT,
    cache_dir: str | None = None,
    skip_existing: bool = True,
) -> dict[str, Any]:
    chosen = list(dialects) if dialects is not None else list(HF_DATASETS)
    unknown = [d for d in chosen if d not in HF_DATASETS]
    if unknown:
        raise ValueError(f"unknown dialect {unknown}; expected {list(HF_DATASETS)}")

    import gc

    # Pass 1 keeps text only. Audio bytes for all five dialects are several
    # GB and were previously held until the last dialect finished indexing,
    # which OOM-killed the process before any FLAC was written.
    raw: list[dict[str, Any]] = []
    for dialect in chosen:
        repo = HF_DATASETS[dialect]
        log.info("indexing %s (%s)", dialect, repo)
        raw.extend(_collect_clips(dialect, repo, cache_dir))
        gc.collect()

    assignment = assign_prompts(r["text"] for r in raw)
    by_dialect: dict[str, list[tuple[int, dict[str, Any]]]] = {d: [] for d in chosen}
    for i, row in enumerate(raw):
        by_dialect[row["dialect"]].append((i, row))
    del raw
    gc.collect()

    buckets: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    for dialect in chosen:
        repo = HF_DATASETS[dialect]
        log.info("writing %s", dialect)
        loaded = {name: ds for name, ds in _load_splits(repo, cache_dir)}
        for n, (i, row) in enumerate(by_dialect[dialect], start=1):
            split = assignment.get(row["text"], "train")
            clip_id = f"leyu_{row['dialect']}_{split}_{i:06d}"
            rel = f"data/leyu_am/{split}/{clip_id}.flac"
            dest = out_dir / rel
            rec = {
                "audio": rel,
                "text": row["text"],
                "text_raw": row["text_raw"],
                "text_eval": row["text_eval"],
                "speaker_id": row["speaker_id"],
                "prompt_id": f"leyu_{row['text'][:48]}",
                "clip_id": clip_id,
                "gender": row["gender"],
                "age_band": "",
                "region": row["region"],
                "duration_s": 0.0,
                "speech_s": 0.0,
                "speech_start_s": 0.0,
                "speech_end_s": 0.0,
                "lufs": None,
                "sample_rate": SAMPLE_RATE,
                "up_votes": 0,
                "down_votes": 0,
                "language": "am",
                "corpus": "leyu",
            }
            if not (skip_existing and dest.exists()):
                audio = loaded[row["hf_split"]][row["hf_index"]]["audio"]
                rec["duration_s"] = write_audio_flac(dest, audio)
                del audio
            else:
                rec["duration_s"] = _duration(dest)
            rec["speech_s"] = rec["duration_s"]
            rec["speech_end_s"] = rec["duration_s"]
            buckets[split].append(rec)
            if n % 500 == 0:
                log.info("  wrote %s %d", dialect, n)
        del loaded
        gc.collect()

    meta_root = out_dir / "metadata" / "leyu_am"
    stats_splits: dict[str, Any] = {}
    for name, rows in buckets.items():
        rows.sort(key=lambda r: r["clip_id"])
        write_jsonl(meta_root / f"{name}.jsonl", rows)
        stats_splits[name] = {
            "clips": len(rows),
            "hours": round(sum(r["duration_s"] for r in rows) / 3600.0, 3),
            "prompts": len({r["text"] for r in rows}),
            "speakers": len({r["speaker_id"] for r in rows}),
            "region": dict(Counter(r["region"] or "(not stated)" for r in rows)),
        }
        log.info("%s %d clips, %.2f h", name, len(rows), stats_splits[name]["hours"])

    # A prompt must not appear in more than one split.
    seen: dict[str, str] = {}
    for name, rows in buckets.items():
        for row in rows:
            prev = seen.get(row["text"])
            if prev is not None and prev != name:
                raise RuntimeError(f"prompt leaked from {prev} into {name}")
            seen[row["text"]] = name

    stats = {
        "datasets": {d: HF_DATASETS[d] for d in chosen},
        "split": "by normalised prompt, seed 0, 5% dev / 5% test",
        "splits": stats_splits,
    }
    save_json(meta_root / "stats.json", stats)
    return stats


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dialects", nargs="+", default=None, choices=list(HF_DATASETS))
    p.add_argument("--out-dir", type=Path, default=ROOT)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--no-skip-existing", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)
    prepare(
        dialects=args.dialects,
        out_dir=args.out_dir,
        cache_dir=args.cache_dir,
        skip_existing=not args.no_skip_existing,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
