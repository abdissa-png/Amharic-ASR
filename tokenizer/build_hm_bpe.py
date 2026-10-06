"""Build a CTC piece vocab using HornMorpho-mined surface affixes + stem BPE.

    python -m tokenizer.build_hm_bpe
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from tokenizer.build_ideal_bpe import CONTINUATION, WORD_INITIAL, build
from tokenizer.hornmorpho_seg import analyze_words, mine, write_stem_corpus
from tokenizer.piece import MARK, PieceTokenizer
from utils import ROOT, setup_logging

log = setup_logging()


def _hm_affixes(mined: dict) -> list[str]:
    """Keep multi-glyph HornMorpho affixes. One-letter locks over-peel stems."""
    out: list[str] = []
    seen: set[str] = set()
    for row in mined["prefixes"]:
        s = row["surface"]
        if re.search(r"[0-9A-Za-z]", s) or len(s) < 2:
            continue
        piece = MARK + s
        if piece not in seen:
            seen.add(piece)
            out.append(piece)
    for row in mined["suffixes"]:
        s = row["surface"]
        if re.search(r"[0-9A-Za-z]", s) or len(s) < 2:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--speech-text", type=Path, default=ROOT / "exp" / "20m_bpe" / "corpus_1024.txt")
    p.add_argument(
        "--word-text",
        type=Path,
        default=ROOT / "exp" / "20m_char_12layer" / "lm_word" / "train.txt",
    )
    p.add_argument("--out-dir", type=Path, default=ROOT / "exp" / "tokenizers" / "hm_bpe")
    p.add_argument("--top-n", type=int, default=2500)
    p.add_argument("--min-affix", type=int, default=15)
    p.add_argument("--max-affix-len", type=int, default=5)
    p.add_argument("--final-size", type=int, default=1024)
    p.add_argument("--candidate-size", type=int, default=2048)
    p.add_argument("--max-piece-len", type=int, default=4)
    args = p.parse_args()

    if not args.speech_text.exists() or not args.word_text.exists():
        log.error("missing speech or word corpus")
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)

    counts = Counter(args.speech_text.read_text(encoding="utf-8").split())
    top = [w for w, _ in counts.most_common(args.top_n)]
    cache = analyze_words(top, args.out_dir / "hm_cache.jsonl")
    mined = mine(cache, counts, min_affix=args.min_affix, max_affix_len=args.max_affix_len)
    stem_info = write_stem_corpus(args.out_dir / "stem_corpus.txt", cache, counts)
    extra = _hm_affixes(mined)
    log.info("HornMorpho extra affixes (%d): %s", len(extra), " ".join(extra[:40]))

    mine_dump = {k: v for k, v in mined.items() if k not in {"prefix_counter", "suffix_counter", "stem_counter"}}
    mine_dump["stem_corpus"] = stem_info
    mine_dump["extra_affixes"] = extra
    (args.out_dir / "hm_mine.json").write_text(json.dumps(mine_dump, ensure_ascii=False, indent=2), encoding="utf-8")

    result = build(
        speech_text=args.speech_text,
        word_text=args.word_text,
        out_dir=args.out_dir,
        final_size=args.final_size,
        candidate_size=args.candidate_size,
        max_piece_len=args.max_piece_len,
        min_speech_affix=10,
        min_speech_merge=20,
        min_word_merge=20,
        min_speech_func=50,
        min_word_func=20,
        word_lines=200_000,
        extra_affixes=extra,
        bpe_text=args.speech_text,
    )
    tok: PieceTokenizer = result["tokenizer"]
    report = result["report"]
    hand = set(WORD_INITIAL + CONTINUATION)
    report["hornmorpho"] = {
        "align_rate": mined["align_rate"],
        "n_surface_aligned": mined["n_surface_aligned"],
        "n_cached": mined["n_cached"],
        "n_stems": mined["n_stems"],
        "extra_affixes": extra,
        "new_vs_hand": [a for a in extra if a not in hand],
    }
    tok.save(args.out_dir / "tokenizer.json")
    (args.out_dir / "vocab.tsv").write_text(
        "\n".join(f"{t}\t{i}" for i, t in enumerate(tok.itos)) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("wrote %s", args.out_dir / "tokenizer.json")
    print("align_rate", mined["align_rate"], "stems", mined["n_stems"])
    print("HM affixes", " ".join(extra[:40]))
    print("layers", json.dumps(report["layers"], ensure_ascii=False))
    print("fertility speech", json.dumps(report["fertility_on_speech"], ensure_ascii=False))
    print("unused", report["unused_on_speech"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
