"""E0 — Dataset validation: statistics, leakage, charset, duration.

Run after ``python prepare_dataset.py``::

    python -m data.audit
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any

from data.normalize import for_training, nfc
from data.prepare import load_split_jsonl
from utils import ROOT, save_json, setup_logging

log = setup_logging()

EXPECTED = {
    "train": {"clips": 13792, "hours": 42.126, "speakers": 360, "prompts": 12453},
    "dev": {"clips": 1526, "hours": 4.699, "speakers": 28, "prompts": 1509},
    "test": {"clips": 1548, "hours": 4.722, "speakers": 105, "prompts": 1526},
}

_MISSING = {"", "nan", "none", "null", "<na>", "n/a"}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().lower() in _MISSING


def _ids(rows: list[dict], key: str) -> set[str]:
    out: set[str] = set()
    for r in rows:
        v = r.get(key)
        if _is_missing(v):
            continue
        out.add(str(v).strip())
    return out


def _overlap_block(a: set[str], b: set[str], *, limit: int = 8) -> dict[str, Any]:
    both = sorted(a & b)
    return {"n": len(both), "examples": both[:limit]}


def leakage_checks(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Speaker/clip disjointness is the E0 hard gate.

    Dataset.ET's published split prompt counts sum to 15,488 while the header
    says 15,443 distinct prompts, so a handful of cross-split sentence matches
    would match the card's own arithmetic. Those are reported, not a hard fail.
    Speakers do add up (360+28+105=493) and *are* a hard fail if they overlap.
    """
    report: dict[str, Any] = {}
    for key, label in (("speaker_id", "speaker"), ("prompt_id", "prompt"), ("clip_id", "clip")):
        sets = {name: _ids(splits[name], key) for name in ("train", "dev", "test")}
        report[f"{label}_n"] = {k: len(v) for k, v in sets.items()}
        report[f"{label}_missing"] = {
            name: sum(1 for r in splits[name] if _is_missing(r.get(key))) for name in splits
        }
        report[f"{label}_overlap_train_dev"] = _overlap_block(sets["train"], sets["dev"])
        report[f"{label}_overlap_train_test"] = _overlap_block(sets["train"], sets["test"])
        report[f"{label}_overlap_dev_test"] = _overlap_block(sets["dev"], sets["test"])

    def sentences(norm) -> dict[str, set[str]]:
        return {
            name: {norm(r.get("text_raw") or r.get("text") or "") for r in rows if not _is_missing(r.get("text_raw") or r.get("text"))}
            for name, rows in splits.items()
        }

    raw = sentences(lambda s: s)
    folded = sentences(lambda s: nfc(s).strip())
    for kind, sets in (("sentence", raw), ("sentence_nfc", folded)):
        report[f"{kind}_overlap_train_dev"] = _overlap_block(sets["train"], sets["dev"])
        report[f"{kind}_overlap_train_test"] = _overlap_block(sets["train"], sets["test"])
        report[f"{kind}_overlap_dev_test"] = _overlap_block(sets["dev"], sets["test"])

    report["speakers_clean"] = (
        report["speaker_overlap_train_dev"]["n"] == 0
        and report["speaker_overlap_train_test"]["n"] == 0
        and report["speaker_overlap_dev_test"]["n"] == 0
    )
    report["clips_clean"] = (
        report["clip_overlap_train_dev"]["n"] == 0
        and report["clip_overlap_train_test"]["n"] == 0
        and report["clip_overlap_dev_test"]["n"] == 0
    )
    report["prompts_clean"] = (
        report["sentence_overlap_train_dev"]["n"] == 0
        and report["sentence_overlap_train_test"]["n"] == 0
        and report["sentence_overlap_dev_test"]["n"] == 0
    )
    # Hard E0 gate: speaker-disjoint + unique clips. Prompt overlap is reported.
    report["clean"] = report["speakers_clean"] and report["clips_clean"]
    return report


def duration_stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    durs = sorted(r["duration_s"] for r in rows)
    if not durs:
        return {}
    n = len(durs)

    def pct(p: float) -> float:
        i = min(n - 1, max(0, int(round(p * (n - 1)))))
        return round(durs[i], 3)

    return {
        "min": round(durs[0], 3),
        "p10": pct(0.10),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "max": round(durs[-1], 3),
        "mean": round(sum(durs) / n, 3),
        "short_lt_4s": sum(1 for d in durs if d < 4.0),
        "long_gt_12s": sum(1 for d in durs if d > 12.0),
    }


def charset_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    chars: Counter[str] = Counter()
    for r in rows:
        chars.update(for_training(r.get("text_raw") or r["text"]))
    return {
        "n_unique": len(chars),
        "chars": "".join(sorted(chars)),
        "counts": dict(chars.most_common()),
    }


def compare_expected(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    mismatches: dict[str, Any] = {}
    for name, exp in EXPECTED.items():
        rows = splits.get(name, [])
        hours = sum(r["duration_s"] for r in rows) / 3600.0
        got = {
            "clips": len(rows),
            "hours": round(hours, 3),
            "speakers": len(_ids(rows, "speaker_id")),
            "prompts": len(_ids(rows, "prompt_id")),
        }
        delta = {}
        for k, v in exp.items():
            if k == "hours":
                if abs(got[k] - v) > 0.05:
                    delta[k] = {"expected": v, "got": got[k]}
            elif got[k] != v:
                delta[k] = {"expected": v, "got": got[k]}
        if delta:
            mismatches[name] = delta
        else:
            mismatches[name] = "ok"
    return mismatches


def audit(meta_dir: Path | None = None) -> dict[str, Any]:
    splits = {name: load_split_jsonl(name, meta_dir) for name in ("train", "dev", "test")}
    report = {
        "leakage": leakage_checks(splits),
        "expected": compare_expected(splits),
        "duration": {name: duration_stats(rows) for name, rows in splits.items()},
        "charset": {name: charset_report(rows) for name, rows in splits.items()},
        "empty_text": {
            name: sum(1 for r in rows if not r["text"].strip()) for name, rows in splits.items()
        },
        "missing_audio": {
            name: sum(1 for r in rows if not (ROOT / r["audio"]).exists())
            for name, rows in splits.items()
        },
    }
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meta-dir", type=Path, default=ROOT / "metadata")
    p.add_argument("--out", type=Path, default=ROOT / "metadata" / "audit.json")
    args = p.parse_args()
    report = audit(args.meta_dir)
    save_json(args.out, report)
    leak = report["leakage"]

    def _n(key: str) -> int:
        return int(leak[key]["n"])

    log.info("expected-size check: %s", report["expected"])
    log.info(
        "speakers: train/dev/test=%s missing=%s overlaps td/tt/dt=%d/%d/%d",
        leak["speaker_n"],
        leak["speaker_missing"],
        _n("speaker_overlap_train_dev"),
        _n("speaker_overlap_train_test"),
        _n("speaker_overlap_dev_test"),
    )
    log.info(
        "clips: overlaps td/tt/dt=%d/%d/%d",
        _n("clip_overlap_train_dev"),
        _n("clip_overlap_train_test"),
        _n("clip_overlap_dev_test"),
    )
    log.info(
        "raw sentences: overlaps td/tt/dt=%d/%d/%d examples_tt=%s",
        _n("sentence_overlap_train_dev"),
        _n("sentence_overlap_train_test"),
        _n("sentence_overlap_dev_test"),
        leak["sentence_overlap_train_test"]["examples"],
    )
    log.info(
        "NFC sentences: overlaps td/tt/dt=%d/%d/%d",
        _n("sentence_nfc_overlap_train_dev"),
        _n("sentence_nfc_overlap_train_test"),
        _n("sentence_nfc_overlap_dev_test"),
    )
    if leak["speakers_clean"] and leak["clips_clean"]:
        if leak["prompts_clean"]:
            log.info("E0 passed: speaker-, clip-, and prompt-disjoint")
        else:
            n_prompt = (
                _n("sentence_overlap_train_dev")
                + _n("sentence_overlap_train_test")
                + _n("sentence_overlap_dev_test")
            )
            log.warning(
                "E0 passed on speakers/clips; %d raw-sentence overlaps across splits "
                "(Dataset.ET split prompt counts already do not add up to the 15,443 header). "
                "Details in %s",
                n_prompt,
                args.out,
            )
        return 0
    log.error("E0 FAILED: speaker or clip leakage. See %s", args.out)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
