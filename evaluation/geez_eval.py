"""Dataset.ET official Ge'ez evaluation normaliser.

Copied from snapwre/amharic-asr-benchmark ``scripts/geez_eval.py`` so WER/CER
figures are comparable with the published Amharic ASR Benchmark.

Scoring only: fold homophone families, strip punctuation/symbols, NFC, casefold.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from typing import Mapping

# Each pair shifts one consonant family onto its homophone base. The Ethiopic
# block lays every consonant out as a contiguous run of vowel orders from a
# fixed base, so shifting the base and keeping the offset folds all orders at
# once — including rare orders a hand-written character list would miss.
_GEEZ_FOLD_BASES = [
    (0x1210, 0x1200),  # ሐ -> ሀ
    (0x1280, 0x1200),  # ኀ -> ሀ
    (0x1220, 0x1230),  # ሠ -> ሰ
    (0x12D0, 0x12A0),  # ዐ -> አ
    (0x1340, 0x1338),  # ፀ -> ጸ
]
_ORDERS = 8

_GEEZ_MAP = {src + o: dst + o for src, dst in _GEEZ_FOLD_BASES for o in range(_ORDERS)}
_WS = re.compile(r"\s+")
_LANG_TAG = re.compile(r"^\s*\[[A-Z]{2,4}\]\s*")


def strip_lang_tag(text: str) -> str:
    """Drop an Ethio-ASR-style ``[AMH]`` prefix before scoring."""
    return _LANG_TAG.sub("", text or "")


def normalize(text: str, *, fold_geez: bool = True) -> str:
    text = unicodedata.normalize("NFC", strip_lang_tag(text or ""))
    if fold_geez:
        text = text.translate(_GEEZ_MAP)
    # Ethiopic punctuation (። ፣ ፤) sits inside the same Unicode block as
    # letters, so a character-range test cannot separate them. Ask Unicode.
    text = "".join(" " if unicodedata.category(c)[0] in ("P", "S") else c for c in text)
    return _WS.sub(" ", text.casefold()).strip()


def evaluate(
    references: Mapping[str, str],
    predictions: Mapping[str, str],
    *,
    fold_geez: bool = True,
) -> dict:
    import jiwer

    missing = [k for k in references if k not in predictions]
    empty = [k for k, v in predictions.items() if not (v or "").strip()]

    ids = list(references)
    refs = [references[i] for i in ids]
    hyps = [predictions.get(i, "") for i in ids]

    def measure(r: list[str], h: list[str]) -> tuple[float, float]:
        pairs = [(a, b) for a, b in zip(r, h, strict=True) if a.strip()]
        if not pairs:
            return float("nan"), float("nan")
        return (
            jiwer.wer([p[0] for p in pairs], [p[1] for p in pairs]),
            jiwer.cer([p[0] for p in pairs], [p[1] for p in pairs]),
        )

    raw_wer, raw_cer = measure(refs, hyps)
    n_wer, n_cer = measure(
        [normalize(r, fold_geez=fold_geez) for r in refs],
        [normalize(h, fold_geez=fold_geez) for h in hyps],
    )
    return {
        "clips": len(ids),
        "wer": round(n_wer * 100, 2),
        "cer": round(n_cer * 100, 2),
        "wer_raw": round(raw_wer * 100, 2),
        "cer_raw": round(raw_cer * 100, 2),
        "missing_predictions": len(missing),
        "empty_predictions": len(empty),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True, help='JSONL of {"clip_id": ..., "text": ...}')
    p.add_argument("--dataset", default="snapwre/amharic-speech")
    p.add_argument("--split", default="test")
    p.add_argument("--no-fold", action="store_true")
    args = p.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split)
    references = {r["clip_id"]: r["sentence"] for r in ds}

    predictions: dict[str, str] = {}
    with open(args.predictions, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                predictions[row["clip_id"]] = row.get("text", "")

    result = evaluate(references, predictions, fold_geez=not args.no_fold)
    print(json.dumps(result, indent=2))
    if result["missing_predictions"]:
        print(
            f"\nWARNING: {result['missing_predictions']} of {result['clips']} test "
            f"clips had no prediction and were scored as full deletions.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
