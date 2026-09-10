"""Download Dataset.ET Amharic Speech and write a frozen local corpus.

Produces::

    data/train/*.flac
    data/dev/*.flac
    data/test/*.flac
    metadata/train.jsonl
    metadata/dev.jsonl
    metadata/test.jsonl
    metadata/stats.json
    metadata/fingerprint.json

The Hugging Face validation split is stored as ``dev``.
``prompt_id`` is a deterministic hash of the prompt text (the release has no
prompt_id column; disjointness is over the ``sentence`` field).
"""

from __future__ import annotations

import argparse
import hashlib
import io
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from data.normalize import for_eval, for_training, nfc
from utils import ROOT, save_json, setup_logging, write_jsonl

HF_DATASET = "snapwre/amharic-speech"
HF_REVISION = "main"  # v0.2.0 is the current default config
SAMPLE_RATE = 16000
SPLITS = {
    "train": "train",
    "validation": "dev",
    "test": "test",
}

log = setup_logging()


def prompt_id(sentence: str) -> str:
    digest = hashlib.sha1(nfc(sentence).encode("utf-8")).hexdigest()[:12]
    return f"prm_{digest}"


def _null(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return text


def example_record(row: dict[str, Any], audio_rel: str) -> dict[str, Any]:
    sentence = row["sentence"] or ""
    return {
        "audio": audio_rel,
        "text": for_training(sentence),
        "text_raw": sentence,
        "text_eval": for_eval(sentence),
        "speaker_id": _null(row.get("speaker_id")),
        "prompt_id": prompt_id(sentence),
        "clip_id": _null(row.get("clip_id")),
        "gender": _null(row.get("gender")),
        "age_band": _null(row.get("age_band")),
        "region": _null(row.get("region")),
        "duration_s": float(row.get("duration_s") or 0.0),
        "speech_s": float(row.get("speech_s") or 0.0),
        "speech_start_s": float(row.get("speech_start_s") or 0.0),
        "speech_end_s": float(row.get("speech_end_s") or 0.0),
        "lufs": None if row.get("lufs") is None else float(row["lufs"]),
        "sample_rate": SAMPLE_RATE,
        "up_votes": int(row.get("up_votes") or 0),
        "down_votes": int(row.get("down_votes") or 0),
        "language": _null(row.get("language") or "am"),
    }


def _write_flac(path: Path, array, sample_rate: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), array, sample_rate, format="FLAC", subtype="PCM_16")


def _as_audio_dict(audio: Any) -> dict[str, Any]:
    if isinstance(audio, dict):
        return audio
    # datasets AudioDecoder / mapping-like objects
    out: dict[str, Any] = {}
    for key in ("bytes", "path", "array", "sampling_rate"):
        if hasattr(audio, key):
            out[key] = getattr(audio, key)
        elif isinstance(audio, dict) or hasattr(audio, "get"):
            try:
                out[key] = audio[key]  # type: ignore[index]
            except Exception:
                pass
    return out


def _audio_array(row: dict[str, Any]):
    import soundfile as sf

    audio = _as_audio_dict(row["audio"])
    if audio.get("array") is not None:
        arr = audio["array"]
        sr = int(audio.get("sampling_rate") or SAMPLE_RATE)
        return arr, sr
    if audio.get("bytes"):
        arr, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
        return arr, sr
    path = audio.get("path")
    if path:
        arr, sr = sf.read(str(path), dtype="float32")
        return arr, sr
    raise TypeError(f"unrecognised audio payload: {type(row.get('audio'))}")


def write_audio_flac(dest: Path, audio: Any, sample_rate: int = SAMPLE_RATE) -> float:
    """Decode an HF audio payload to 16 kHz mono FLAC. Returns duration in seconds."""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = _as_audio_dict(audio) if not isinstance(audio, dict) else audio
    if payload.get("bytes"):
        arr, sr = sf.read(io.BytesIO(payload["bytes"]), dtype="float32")
    elif payload.get("array") is not None:
        arr = np.asarray(payload["array"], dtype=np.float32)
        sr = int(payload.get("sampling_rate") or sample_rate)
    elif payload.get("path"):
        arr, sr = sf.read(str(payload["path"]), dtype="float32")
    else:
        raise TypeError(f"unrecognised audio payload: {type(audio)}")
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(axis=1)
    sr = int(sr)
    if sr != sample_rate:
        wav = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)
        arr = torchaudio.functional.resample(wav, sr, sample_rate).squeeze(0).numpy()
        sr = sample_rate
    sf.write(str(dest), arr, sr, format="FLAC", subtype="PCM_16")
    return float(arr.shape[0]) / float(sr)


def _materialize_audio(dest: Path, row: dict[str, Any]) -> None:
    """Write 16 kHz FLAC. Prefer raw bytes — Dataset.ET already ships FLAC."""
    import shutil

    dest.parent.mkdir(parents=True, exist_ok=True)
    audio = _as_audio_dict(row.get("audio"))
    data = audio.get("bytes")
    if data:
        dest.write_bytes(data)
        return
    src = audio.get("path")
    if src and Path(src).exists():
        shutil.copyfile(src, dest)
        return
    arr, sr = _audio_array(row)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{row.get('clip_id')}: expected {SAMPLE_RATE} Hz, got {sr}")
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(axis=1)
    _write_flac(dest, arr, SAMPLE_RATE)


def _iter_hf_split(split: str, cache_dir: str | None):
    """Iterate rows without torchcodec. datasets 5+ requires it if decode=True."""
    from datasets import Audio, load_dataset

    ds = load_dataset(
        HF_DATASET,
        split=split,
        cache_dir=cache_dir,
        revision=HF_REVISION,
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE, decode=False))
    for row in ds:
        yield dict(row)


def prepare(
    *,
    out_dir: Path = ROOT,
    cache_dir: str | None = None,
    metadata_only: bool = False,
    skip_existing: bool = True,
    max_per_split: int | None = None,
) -> dict[str, Any]:
    meta_root = out_dir / "metadata"
    meta_root.mkdir(parents=True, exist_ok=True)

    all_rows: dict[str, list[dict[str, Any]]] = {}
    for hf_split, local_split in SPLITS.items():
        log.info("loading %s → %s", hf_split, local_split)
        records: list[dict[str, Any]] = []
        n = 0
        for row in _iter_hf_split(hf_split, cache_dir):
            clip_id = row["clip_id"]
            rel = f"data/{local_split}/{clip_id}.flac"
            rec = example_record(row, rel)
            if not metadata_only:
                dest = out_dir / rel
                if not (skip_existing and dest.exists()):
                    _materialize_audio(dest, row)
            records.append(rec)
            n += 1
            if n % 500 == 0:
                log.info("  %s %d", local_split, n)
            if max_per_split and n >= max_per_split:
                break
        records.sort(key=lambda r: r["clip_id"])
        write_jsonl(meta_root / f"{local_split}.jsonl", records)
        all_rows[local_split] = records
        log.info("wrote %s (%d clips)", local_split, len(records))

    stats = corpus_stats(all_rows)
    fingerprint = corpus_fingerprint(all_rows)
    save_json(meta_root / "stats.json", stats)
    save_json(meta_root / "fingerprint.json", fingerprint)
    log.info("hours: %s", {k: v["hours"] for k, v in stats["splits"].items()})
    return {"stats": stats, "fingerprint": fingerprint}


def corpus_stats(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {"dataset": HF_DATASET, "revision": HF_REVISION, "splits": {}}
    for name, rows in splits.items():
        speakers = {r["speaker_id"] for r in rows}
        prompts = {r["prompt_id"] for r in rows}
        hours = sum(r["duration_s"] for r in rows) / 3600.0
        out["splits"][name] = {
            "clips": len(rows),
            "hours": round(hours, 3),
            "speakers": len(speakers),
            "prompts": len(prompts),
            "gender": dict(Counter(r["gender"] or "(not stated)" for r in rows)),
            "age_band": dict(Counter(r["age_band"] or "(not stated)" for r in rows)),
            "region": dict(Counter(r["region"] or "(not stated)" for r in rows)),
        }
    return out


def corpus_fingerprint(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Stable hash of clip ids + training targets so the processed set is auditable."""
    fp = hashlib.sha256()
    for name in sorted(splits):
        for row in splits[name]:
            fp.update(f"{name}|{row['clip_id']}|{row['text']}\n".encode())
    return {
        "sha256": fp.hexdigest(),
        "n_train": len(splits.get("train", [])),
        "n_dev": len(splits.get("dev", [])),
        "n_test": len(splits.get("test", [])),
    }


def load_split_jsonl(split: str, meta_dir: Path | None = None) -> list[dict[str, Any]]:
    meta_dir = meta_dir or (ROOT / "metadata")
    path = Path(meta_dir) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run python prepare_dataset.py first")
    from utils import load_jsonl

    return load_jsonl(path)


def metadata_dir(cfg: dict[str, Any] | None = None) -> Path:
    """``paths.metadata_dir`` from a train config, else ``metadata/``."""
    if cfg:
        return ROOT / cfg.get("paths", {}).get("metadata_dir", "metadata")
    return ROOT / "metadata"


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=ROOT)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--metadata-only", action="store_true", help="write jsonl without extracting audio")
    p.add_argument("--no-skip-existing", action="store_true")
    p.add_argument("--max-per-split", type=int, default=None, help="debug: cap clips per split")
    args = p.parse_args(list(argv) if argv is not None else None)
    prepare(
        out_dir=args.out_dir,
        cache_dir=args.cache_dir,
        metadata_only=args.metadata_only,
        skip_existing=not args.no_skip_existing,
        max_per_split=args.max_per_split,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
