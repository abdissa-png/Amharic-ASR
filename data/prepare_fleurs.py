"""Download Google FLEURS Amharic (am_et) and freeze it like Dataset.ET.

Source: https://huggingface.co/datasets/google/fleurs/tree/main/parquet-data/am_et

    python -m data.prepare_fleurs              # test + validation (~0.5 GB)
    python -m data.prepare_fleurs --splits test validation train   # +2.5 GB train

Writes::

    data/fleurs_am_et/{train,dev,test}/*.flac
    metadata/fleurs_am_et/{train,dev,test}.jsonl
    metadata/fleurs_am_et/stats.json

FLEURS ASR scoring uses the normalised ``transcription`` field (not
``raw_transcription``). We still run Dataset.ET Ge'ez eval folding on top
so WER/CER is comparable with our other numbers.
"""

from __future__ import annotations

import argparse
import io
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from data.normalize import for_eval, for_training
from utils import ROOT, save_json, setup_logging, write_jsonl

HF_REPO = "google/fleurs"
HF_PREFIX = "parquet-data/am_et"
SAMPLE_RATE = 16000
# Hugging Face validation → local "dev", matching Dataset.ET.
SPLIT_FILES = {
    "train": "train-00000-of-00001.parquet",
    "validation": "validation-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}
SPLIT_LOCAL = {"train": "train", "validation": "dev", "test": "test"}

log = setup_logging()


def _download_parquet(split: str, cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    name = f"{HF_PREFIX}/{SPLIT_FILES[split]}"
    log.info("downloading %s", name)
    path = hf_hub_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        filename=name,
        local_dir=str(cache_dir),
    )
    return Path(path)


def _audio_from_cell(cell: Any) -> tuple[Any, int | None]:
    """Return (array or bytes, sampling_rate)."""
    if cell is None:
        raise ValueError("missing audio")
    if isinstance(cell, dict):
        sr = cell.get("sampling_rate")
        if cell.get("bytes"):
            return cell["bytes"], sr
        if cell.get("array") is not None:
            return cell["array"], sr
        if cell.get("path"):
            return Path(cell["path"]), sr
    if hasattr(cell, "as_py"):
        return _audio_from_cell(cell.as_py())
    raise TypeError(f"unrecognised FLEURS audio cell: {type(cell)}")


def _write_audio(dest: Path, payload: Any, sampling_rate: int | None) -> float:
    import numpy as np
    import soundfile as sf

    dest.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, (bytes, bytearray, memoryview)):
        arr, sr = sf.read(io.BytesIO(bytes(payload)), dtype="float32")
    elif isinstance(payload, Path):
        arr, sr = sf.read(str(payload), dtype="float32")
    else:
        arr = np.asarray(payload, dtype=np.float32)
        sr = int(sampling_rate or SAMPLE_RATE)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(axis=1)
    sr = int(sr)
    if sr != SAMPLE_RATE:
        import torch
        import torchaudio

        wav = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)
        arr = torchaudio.functional.resample(wav, sr, SAMPLE_RATE).squeeze(0).numpy()
        sr = SAMPLE_RATE
    sf.write(str(dest), arr, sr, format="FLAC", subtype="PCM_16")
    return float(arr.shape[0]) / float(sr)


def _gender(value: Any) -> str:
    names = {0: "male", 1: "female", 2: "other"}
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return names.get(int(value), str(value))
    except (TypeError, ValueError):
        return str(value)


def prepare(
    *,
    splits: Iterable[str] = ("test", "validation"),
    out_dir: Path = ROOT,
    cache_dir: Path | None = None,
    skip_existing: bool = True,
) -> dict[str, Any]:
    cache_dir = cache_dir or (ROOT / ".cache" / "fleurs_am_et")
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_root = out_dir / "metadata" / "fleurs_am_et"
    meta_root.mkdir(parents=True, exist_ok=True)

    import pyarrow.parquet as pq

    all_rows: dict[str, list[dict[str, Any]]] = {}
    for hf_split in splits:
        if hf_split not in SPLIT_FILES:
            raise ValueError(f"unknown split {hf_split}; expected {list(SPLIT_FILES)}")
        local = SPLIT_LOCAL[hf_split]
        parquet_path = _download_parquet(hf_split, cache_dir)
        table = pq.read_table(parquet_path)
        log.info("loaded %s (%d rows, %s)", hf_split, table.num_rows, parquet_path)
        records: list[dict[str, Any]] = []
        cols = {name: table.column(name) for name in table.column_names}
        for i in range(table.num_rows):
            src_id = int(cols["id"][i].as_py()) if "id" in cols else i
            clip_id = f"fleurs_am_{local}_{src_id:05d}_{i:04d}"
            rel = f"data/fleurs_am_et/{local}/{clip_id}.flac"
            dest = out_dir / rel
            transcription = cols["transcription"][i].as_py() if "transcription" in cols else ""
            raw = cols["raw_transcription"][i].as_py() if "raw_transcription" in cols else transcription
            num_samples = cols["num_samples"][i].as_py() if "num_samples" in cols else None
            duration = float(num_samples) / SAMPLE_RATE if num_samples else 0.0
            if not (skip_existing and dest.exists()):
                payload, sr = _audio_from_cell(cols["audio"][i].as_py() if "audio" in cols else None)
                duration = _write_audio(dest, payload, sr)
            elif not duration:
                import soundfile as sf

                info = sf.info(str(dest))
                duration = float(info.frames) / float(info.samplerate)
            gender = _gender(cols["gender"][i].as_py() if "gender" in cols else None)
            rec = {
                "audio": rel,
                "text": for_training(transcription or ""),
                "text_raw": transcription or "",
                "text_eval": for_eval(transcription or ""),
                "text_raw_fleurs": raw or "",
                "speaker_id": "",
                "prompt_id": f"fleurs_{src_id:05d}",
                "clip_id": clip_id,
                "gender": gender,
                "age_band": "",
                "region": "",
                "duration_s": duration,
                "speech_s": duration,
                "speech_start_s": 0.0,
                "speech_end_s": duration,
                "lufs": None,
                "sample_rate": SAMPLE_RATE,
                "up_votes": 0,
                "down_votes": 0,
                "language": "am",
                "fleurs_id": src_id,
            }
            records.append(rec)
            if (i + 1) % 100 == 0:
                log.info("  %s %d/%d", local, i + 1, table.num_rows)
        records.sort(key=lambda r: r["clip_id"])
        write_jsonl(meta_root / f"{local}.jsonl", records)
        all_rows[local] = records
        log.info("wrote %s (%d clips, %.2f h)", local, len(records), sum(r["duration_s"] for r in records) / 3600.0)

    stats = {
        "dataset": HF_REPO,
        "config": "am_et",
        "source": f"https://huggingface.co/datasets/{HF_REPO}/tree/main/{HF_PREFIX}",
        "splits": {},
    }
    for name, rows in all_rows.items():
        stats["splits"][name] = {
            "clips": len(rows),
            "hours": round(sum(r["duration_s"] for r in rows) / 3600.0, 3),
            "prompts": len({r["prompt_id"] for r in rows}),
            "gender": dict(Counter(r["gender"] or "(not stated)" for r in rows)),
        }
    save_json(meta_root / "stats.json", stats)
    return stats


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--splits", nargs="+", default=["test", "validation"], choices=list(SPLIT_FILES))
    p.add_argument("--out-dir", type=Path, default=ROOT)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--no-skip-existing", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)
    prepare(
        splits=args.splits,
        out_dir=args.out_dir,
        cache_dir=args.cache_dir,
        skip_existing=not args.no_skip_existing,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
