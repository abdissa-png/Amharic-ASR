"""Train set for the Leyu continuation: screened pool plus Leyu train.

Dev and test stay the Dataset.ET splits in ``metadata/pool_clean``, so the
number logged each epoch is still comparable to the pool's 30.5% CTC word
error. Leyu dev and test are left in ``metadata/leyu_am`` and scored with
``eval_leyu.py``.

    python -m data.prepare_pool_leyu
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from utils import ROOT, load_jsonl, save_json, setup_logging, write_jsonl

log = setup_logging()


def prepare(
    *,
    out_dir: Path = ROOT,
    pool_dir: Path | None = None,
    leyu_dir: Path | None = None,
) -> dict:
    pool = pool_dir or (out_dir / "metadata" / "pool_clean")
    leyu = leyu_dir or (out_dir / "metadata" / "leyu_am")
    pool_train = pool / "train.jsonl"
    leyu_train = leyu / "train.jsonl"
    missing = [str(p) for p in (pool_train, leyu_train, pool / "dev.jsonl", pool / "test.jsonl") if not p.exists()]
    if missing:
        raise FileNotFoundError("missing jsonl: " + ", ".join(missing))

    base = load_jsonl(pool_train)
    extra = load_jsonl(leyu_train)
    ids = [r.get("clip_id") for r in base]
    ids.extend(r.get("clip_id") for r in extra)
    dup = [cid for cid, n in Counter(ids).items() if cid and n > 1]
    if dup:
        raise RuntimeError(f"{len(dup)} clip_ids appear in both train sets, e.g. {dup[:3]}")

    train = base + extra
    train.sort(key=lambda r: r.get("clip_id") or "")
    dest = out_dir / "metadata" / "pool_leyu"
    dest.mkdir(parents=True, exist_ok=True)
    write_jsonl(dest / "train.jsonl", train)
    for name in ("dev.jsonl", "test.jsonl"):
        write_jsonl(dest / name, load_jsonl(pool / name))

    hours = sum(float(r.get("duration_s") or 0.0) for r in train) / 3600.0
    stats = {
        "train_clips": len(train),
        "train_hours": round(hours, 3),
        "pool_clips": len(base),
        "leyu_clips": len(extra),
        "corpus": dict(Counter(r.get("corpus") or "?" for r in train)),
        "dev": "pool_clean dev (Dataset.ET)",
        "test": "pool_clean test (Dataset.ET)",
    }
    save_json(dest / "stats.json", stats)
    log.info(
        "pool_leyu train=%.2fh (%d clips, %d from leyu)",
        hours,
        len(train),
        len(extra),
    )
    return stats


def main() -> int:
    prepare()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
