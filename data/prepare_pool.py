"""Merge prepared Amharic corpora into one training jsonl.

Does not download audio. Run the per-corpus prepare scripts first::

    python prepare_dataset.py
    python -m data.prepare_fleurs --splits train validation test
    python -m data.prepare_alffa
    python -m data.prepare_bdu
    python -m data.prepare_leyu
    python -m data.prepare_pool

Writes::

    metadata/pool/train.jsonl     # all sources that exist
    metadata/pool/dev.jsonl       # Dataset.ET dev only (default) so WER stays comparable
    metadata/pool/test.jsonl      # Dataset.ET test only
    metadata/pool/dev_pooled.jsonl
    metadata/pool/test_pooled.jsonl
    metadata/pool/stats.json

Missing sources are skipped. Each row gets ``corpus`` (and a default
``region`` when the source jsonl has none).
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from utils import ROOT, load_jsonl, save_json, setup_logging, write_jsonl

log = setup_logging()

SOURCES: dict[str, dict[str, Any]] = {
    "dataset_et": {
        "train": "metadata/train.jsonl",
        "dev": "metadata/dev.jsonl",
        "test": "metadata/test.jsonl",
        "region": "",
    },
    "fleurs": {
        "train": "metadata/fleurs_am_et/train.jsonl",
        "dev": "metadata/fleurs_am_et/dev.jsonl",
        "test": "metadata/fleurs_am_et/test.jsonl",
        "region": "",
    },
    "alffa": {
        "train": "metadata/alffa_am/train.jsonl",
        "dev": "metadata/alffa_am/dev.jsonl",
        "test": "metadata/alffa_am/test.jsonl",
        "region": "",
    },
    "bdu": {
        "train": "metadata/bdu_am/train.jsonl",
        "dev": "metadata/bdu_am/dev.jsonl",
        "test": "metadata/bdu_am/test.jsonl",
        "region": "sidama",
    },
    "cv": {
        "train": "metadata/cv_am/train.jsonl",
        "dev": "metadata/cv_am/dev.jsonl",
        "test": "metadata/cv_am/test.jsonl",
        "region": "",
    },
    "leyu": {
        "train": "metadata/leyu_am/train.jsonl",
        "dev": "metadata/leyu_am/dev.jsonl",
        "test": "metadata/leyu_am/test.jsonl",
        "region": "",
    },
}


def _stamp(row: dict[str, Any], corpus: str, region_default: str) -> dict[str, Any]:
    out = copy.deepcopy(row)
    out["corpus"] = out.get("corpus") or corpus
    if not out.get("region"):
        out["region"] = region_default
    return out


def _hours(rows: list[dict[str, Any]]) -> float:
    return sum(float(r.get("duration_s") or 0.0) for r in rows) / 3600.0


def prepare(
    *,
    out_dir: Path = ROOT,
    sources: Iterable[str] | None = None,
    dev_policy: str = "et",
) -> dict[str, Any]:
    names = list(sources) if sources is not None else list(SOURCES)
    loaded: dict[str, dict[str, list[dict[str, Any]]]] = {}
    skipped: list[str] = []
    for name in names:
        if name not in SOURCES:
            raise ValueError(f"unknown source {name}; expected {list(SOURCES)}")
        spec = SOURCES[name]
        splits: dict[str, list[dict[str, Any]]] = {}
        missing = False
        for split in ("train", "dev", "test"):
            path = out_dir / spec[split]
            if not path.exists():
                missing = True
                break
            splits[split] = [_stamp(r, name, spec.get("region") or "") for r in load_jsonl(path)]
        if missing:
            skipped.append(name)
            log.warning("skip %s (missing jsonl under %s)", name, spec["train"])
            continue
        loaded[name] = splits
        log.info(
            "%s  train=%.2fh  dev=%.2fh  test=%.2fh",
            name,
            _hours(splits["train"]),
            _hours(splits["dev"]),
            _hours(splits["test"]),
        )

    if "dataset_et" not in loaded and dev_policy == "et":
        log.warning("dev_policy=et but Dataset.ET jsonl is missing; falling back to pooled dev")
        dev_policy = "pooled"

    train: list[dict[str, Any]] = []
    for name, splits in loaded.items():
        train.extend(splits["train"])
    train.sort(key=lambda r: r.get("clip_id") or "")

    pooled_dev: list[dict[str, Any]] = []
    pooled_test: list[dict[str, Any]] = []
    for splits in loaded.values():
        pooled_dev.extend(splits["dev"])
        pooled_test.extend(splits["test"])
    pooled_dev.sort(key=lambda r: r.get("clip_id") or "")
    pooled_test.sort(key=lambda r: r.get("clip_id") or "")

    if dev_policy == "et":
        dev = list(loaded["dataset_et"]["dev"])
        test = list(loaded["dataset_et"]["test"])
    elif dev_policy == "pooled":
        dev, test = pooled_dev, pooled_test
    else:
        raise ValueError("dev_policy must be 'et' or 'pooled'")

    meta_root = out_dir / "metadata" / "pool"
    write_jsonl(meta_root / "train.jsonl", train)
    write_jsonl(meta_root / "dev.jsonl", dev)
    write_jsonl(meta_root / "test.jsonl", test)
    write_jsonl(meta_root / "dev_pooled.jsonl", pooled_dev)
    write_jsonl(meta_root / "test_pooled.jsonl", pooled_test)

    by_corpus = {
        name: {
            "train_hours": round(_hours(splits["train"]), 3),
            "train_clips": len(splits["train"]),
            "dev_clips": len(splits["dev"]),
            "test_clips": len(splits["test"]),
        }
        for name, splits in loaded.items()
    }
    stats = {
        "sources": sorted(loaded),
        "skipped": skipped,
        "dev_policy": dev_policy,
        "train": {
            "clips": len(train),
            "hours": round(_hours(train), 3),
            "corpus": dict(Counter(r.get("corpus") or "?" for r in train)),
            "region": dict(Counter(r.get("region") or "(not stated)" for r in train)),
        },
        "dev": {"clips": len(dev), "hours": round(_hours(dev), 3)},
        "test": {"clips": len(test), "hours": round(_hours(test), 3)},
        "by_corpus": by_corpus,
    }
    save_json(meta_root / "stats.json", stats)
    log.info(
        "pool train=%.2fh (%d clips) from %s; dev_policy=%s",
        stats["train"]["hours"],
        stats["train"]["clips"],
        ",".join(stats["sources"]),
        dev_policy,
    )
    return stats


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=ROOT)
    p.add_argument("--sources", nargs="+", default=None, choices=list(SOURCES))
    p.add_argument(
        "--dev-policy",
        default="et",
        choices=["et", "pooled"],
        help="et = Dataset.ET dev/test (comparable to 28.75 WER). pooled = concat all.",
    )
    args = p.parse_args(list(argv) if argv is not None else None)
    prepare(out_dir=args.out_dir, sources=args.sources, dev_policy=args.dev_policy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
