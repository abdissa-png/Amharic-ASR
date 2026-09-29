"""Teacher fine-tuning entry point (Phase 13).

Fine-tunes Omnilingual ASR CTC ~300M on Dataset.ET train. The actual
Omnilingual training loop lives in that project's recipes; this module
documents the interface and writes a manifest the student can consume.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data.prepare import load_split_jsonl
from utils import ROOT, save_json, setup_logging

log = setup_logging()


def write_manifest(out: Path) -> Path:
    rows = []
    for split in ("train", "dev", "test"):
        for r in load_split_jsonl(split):
            rows.append(
                {
                    "split": split,
                    "audio": r["audio"],
                    "text": r["text"],
                    "clip_id": r["clip_id"],
                }
            )
    save_json(out, {"n": len(rows), "rows": rows[:3], "note": "full jsonl next to this file"})
    # full list as jsonl
    from utils import write_jsonl

    write_jsonl(out.with_suffix(".jsonl"), rows)
    log.info("wrote teacher manifest %s (%d utt)", out.with_suffix(".jsonl"), len(rows))
    return out.with_suffix(".jsonl")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "exp" / "teacher" / "manifest.json")
    args = p.parse_args()
    write_manifest(args.out)
    log.info(
        "Fine-tune omniASR_CTC_300M_v2 with this manifest using the "
        "facebookresearch/omnilingual-asr training recipe, then point "
        "distillation at the resulting checkpoint."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
