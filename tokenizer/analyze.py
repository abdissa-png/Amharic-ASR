"""Tokenizer diagnostics: fertility, OOV pieces, coverage."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from data.prepare import load_split_jsonl
from tokenizer.train_bpe import SpmTokenizer
from utils import ROOT, save_json, setup_logging

log = setup_logging()


def analyse(model_path: Path, split: str = "train") -> dict:
    tok = SpmTokenizer(model_path)
    rows = load_split_jsonl(split)
    n_words = n_pieces = 0
    lens: Counter[int] = Counter()
    for r in rows:
        words = r["text"].split()
        n_words += len(words)
        ids = tok.encode(r["text"])
        n_pieces += len(ids)
        lens[len(ids)] += 1
    return {
        "vocab_size": tok.vocab_size,
        "n_words": n_words,
        "n_pieces": n_pieces,
        "pieces_per_word": round(n_pieces / max(1, n_words), 3),
        "utt_len_p50": _pct(lens, 0.5),
        "utt_len_p90": _pct(lens, 0.9),
    }


def _pct(counter: Counter[int], p: float) -> int:
    items = sorted(counter.elements())
    if not items:
        return 0
    return items[min(len(items) - 1, int(p * (len(items) - 1)))]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    report = analyse(args.model, args.split)
    log.info("%s", report)
    if args.out:
        save_json(args.out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
