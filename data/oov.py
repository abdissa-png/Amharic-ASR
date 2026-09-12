"""Word / OOV analysis on Dataset.ET.

Builds ``V_train`` from training transcripts, then classifies every test word
as SEEN or UNSEEN and further buckets unseen words:

* morphological variants (affix stripping yields a train word)
* compounds (concatenation of two train words)
* proper names (titles, Latin capitals, mixed script)
* loan words (Latin / mixed script, or a small loan lexicon)
* rare (unseen hapaxes in the evaluated split)
* ordinary lexical items (none of the above)

Usage::

    python -m data.oov
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from data.normalize import tokenize_words
from data.prepare import load_split_jsonl
from phonology.decomposition import likely_morphological_variant, split_compound
from utils import ROOT, save_json, setup_logging

log = setup_logging()

LATIN = re.compile(r"[A-Za-z]")
TITLES = {
    "አቶ",
    "ወይዘሮ",
    "ወ/ሮ",
    "ወ/ሪት",
    "ዶክተር",
    "ዶ/ር",
    "ፕሮፌሰር",
    "ኢንጂነር",
    "አምባሳደር",
    "ሚኒስትር",
    "ጠቅላይ",
}


def vocab_from_rows(rows: Iterable[dict[str, Any]], *, policy: str = "eval") -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        src = row.get("text_raw") or row["text"]
        counts.update(tokenize_words(src, policy=policy))
    return counts


def classify_unseen(word: str, v_train: set[str], test_freq: int) -> str:
    if LATIN.search(word):
        if word[:1].isupper() or any(c.isupper() for c in word):
            return "proper_name"
        return "loan_word"
    if likely_morphological_variant(word, v_train):
        return "morphological_variant"
    if split_compound(word, v_train) is not None:
        return "compound"
    if test_freq == 1:
        return "rare"
    return "ordinary"


def utterance_buckets(
    rows: list[dict[str, Any]],
    v_train: set[str],
    *,
    short_s: float = 4.0,
    long_s: float = 12.0,
) -> dict[str, list[str]]:
    """Map analysis bucket → clip_ids."""
    buckets: dict[str, list[str]] = defaultdict(list)
    test_counts = vocab_from_rows(rows)
    for row in rows:
        words = tokenize_words(row.get("text_raw") or row["text"])
        unseen = [w for w in words if w not in v_train]
        clip = row["clip_id"]
        buckets["ALL"].append(clip)
        if unseen:
            buckets["unseen_word_utterances"].append(clip)
        else:
            buckets["familiar_word_utterances"].append(clip)
        if any(test_counts[w] == 1 and w not in v_train for w in words):
            buckets["rare_word_utterances"].append(clip)
        dur = float(row.get("duration_s") or 0.0)
        if dur < short_s:
            buckets["short_utterances"].append(clip)
        if dur > long_s:
            buckets["long_utterances"].append(clip)
    return dict(buckets)


def analyse(meta_dir: Path | None = None) -> dict[str, Any]:
    train = load_split_jsonl("train", meta_dir)
    test = load_split_jsonl("test", meta_dir)
    dev = load_split_jsonl("dev", meta_dir)
    v_counts = vocab_from_rows(train)
    v_train = set(v_counts)
    test_counts = vocab_from_rows(test)

    unseen = {w: c for w, c in test_counts.items() if w not in v_train}
    classes: dict[str, list[str]] = defaultdict(list)
    for w, c in sorted(unseen.items(), key=lambda kv: (-kv[1], kv[0])):
        classes[classify_unseen(w, v_train, c)].append(w)

    report = {
        "v_train_size": len(v_train),
        "v_test_size": len(test_counts),
        "v_dev_size": len(vocab_from_rows(dev)),
        "n_test_tokens": int(sum(test_counts.values())),
        "n_unseen_types": len(unseen),
        "n_unseen_tokens": int(sum(unseen.values())),
        "unseen_token_rate": round(sum(unseen.values()) / max(1, sum(test_counts.values())), 4),
        "unseen_class_counts": {k: len(v) for k, v in classes.items()},
        "unseen_class_examples": {k: v[:25] for k, v in classes.items()},
        "test_buckets": {k: len(v) for k, v in utterance_buckets(test, v_train).items()},
        "dev_buckets": {k: len(v) for k, v in utterance_buckets(dev, v_train).items()},
    }
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meta-dir", type=Path, default=ROOT / "metadata")
    p.add_argument("--out", type=Path, default=ROOT / "metadata" / "oov.json")
    args = p.parse_args()
    report = analyse(args.meta_dir)
    save_json(args.out, report)
    log.info(
        "V_train=%d unseen_types=%d unseen_token_rate=%.3f buckets=%s",
        report["v_train_size"],
        report["n_unseen_types"],
        report["unseen_token_rate"],
        report["test_buckets"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
