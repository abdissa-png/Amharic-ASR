"""Train SentencePiece BPE on speech + word corpus and compare to the unigram vocabs.

Unigram EM can skip characters and keep whole words. BPE is forced to start
from the character inventory and spend leftover slots on frequent merges
(prefixes / suffixes / stems).

    python -m tokenizer.compare_bpe
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from tokenizer.compare_word_spm import (
    _ETH,
    _SPECIAL,
    _fertility,
    _pieces,
    _sample_lines,
    _stats,
    train_from_file,
)
from tokenizer.train_bpe import SpmTokenizer
from utils import ROOT, setup_logging

log = setup_logging()

# Productive Amharic affixes we want as *atomic* pieces, not only buried in words.
_WORD_INITIAL = [
    "▁የ",
    "▁በ",
    "▁ለ",
    "▁ከ",
    "▁ወደ",
    "▁ስለ",
    "▁እንደ",
    "▁እስከ",
    "▁ያለ",
    "▁አል",
    "▁አይ",
    "▁ይ",
    "▁ት",
    "▁ን",
    "▁መ",
    "▁ተ",
    "▁አስ",
    "▁እየ",
    "▁የሚ",
    "▁የተ",
    "▁በመ",
    "▁ለማ",
    "▁እንዲ",
    "▁እንደሚ",
    "▁እንዳ",
    "▁አንድ",
]
_CONTINUATION = [
    "ዎች",
    "ነት",
    "ዋል",
    "ናል",
    "ቸው",
    "ችው",
    "አችው",
    "ዋ",
    "ው",
    "ን",
    "ም",
    "ል",
    "ች",
    "ት",
]


def _ensure_model(text_path: Path, *, vocab_size: int, model_type: str, prefix: Path) -> Path:
    model = Path(str(prefix) + ".model")
    if model.exists():
        log.info("reusing %s", model)
        return model
    log.info("training %s on %s", model_type, text_path)
    path = train_from_file(
        text_path,
        vocab_size=vocab_size,
        model_type=model_type,
        out_prefix=prefix,
    )
    log.info("wrote %s", path)
    return path


def _body(piece: str) -> str:
    return piece[1:] if piece.startswith("▁") else piece


def _length_hist(pieces: list[str]) -> dict[str, int]:
    hist: Counter[int] = Counter()
    for p in pieces:
        if p in _SPECIAL:
            continue
        hist[len(_body(p)) or 1] += 1
    return {str(k): hist[k] for k in sorted(hist)}


def _atomic_and_buried(pieces: list[str]) -> tuple[set[str], set[str]]:
    atomic: set[str] = set()
    buried: set[str] = set()
    for p in pieces:
        if p in _SPECIAL:
            continue
        body = _body(p)
        fidel = [c for c in body if _ETH.search(c)]
        if len(body) == 1 and fidel:
            atomic.add(body)
        else:
            buried.update(fidel)
    return atomic, buried


def _corpus_fidel(path: Path, max_chars: int = 5_000_000) -> set[str]:
    seen: set[str] = set()
    n = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            for c in line:
                if _ETH.search(c):
                    seen.add(c)
                n += 1
                if n >= max_chars:
                    return seen
    return seen


def _char_coverage(pieces: list[str], corpus_chars: set[str]) -> dict:
    atomic, buried = _atomic_and_buried(pieces)
    in_any = atomic | buried
    missing = sorted(corpus_chars - in_any)
    buried_only = sorted(corpus_chars & (buried - atomic))
    return {
        "corpus_fidel": len(corpus_chars),
        "atomic_fidel_pieces": len(atomic),
        "atomic_covering_corpus": len(atomic & corpus_chars),
        "buried_only": len(buried_only),
        "missing_from_vocab": len(missing),
        "buried_only_examples": buried_only[:30],
        "missing_examples": missing[:30],
        "pct_corpus_atomic": round(100.0 * len(atomic & corpus_chars) / max(1, len(corpus_chars)), 1),
    }


def _affix_hits(piece_set: set[str]) -> dict:
    initial = [a for a in _WORD_INITIAL if a in piece_set]
    cont = [a for a in _CONTINUATION if a in piece_set]
    return {
        "word_initial_present": initial,
        "word_initial_missing": [a for a in _WORD_INITIAL if a not in piece_set],
        "n_word_initial": f"{len(initial)}/{len(_WORD_INITIAL)}",
        "continuation_present": cont,
        "n_continuation": f"{len(cont)}/{len(_CONTINUATION)}",
    }


def _prefix_inventory(pieces: list[str], *, min_len: int = 2, max_len: int = 4) -> list[str]:
    """Word-initial pieces of typical prefix length, longest first then alpha."""
    out = []
    for p in pieces:
        if p in _SPECIAL or not p.startswith("▁"):
            continue
        n = len(_body(p))
        if min_len <= n <= max_len:
            out.append(p)
    return sorted(out, key=lambda s: (-len(_body(s)), s))


def _continuation_inventory(pieces: list[str], *, min_len: int = 2, max_len: int = 3) -> list[str]:
    out = []
    for p in pieces:
        if p in _SPECIAL or p.startswith("▁"):
            continue
        n = len(p)
        if min_len <= n <= max_len and _ETH.search(p) and not re.search(r"[A-Za-z0-9]", p):
            out.append(p)
    return sorted(out, key=lambda s: (-len(s), s))


def _overlap(a: list[str], b: list[str]) -> dict:
    sa, sb = set(a), set(b)
    both = sa & sb
    return {
        "shared": len(both),
        "a_only": len(sa - sb),
        "b_only": len(sb - sa),
        "jaccard": round(len(both) / max(1, len(sa | sb)), 3),
    }


def _profile(model: Path, corpus: Path, corpus_chars: set[str]) -> dict:
    pieces = [p for p, _ in _pieces(model.with_suffix(".vocab"))]
    tok = SpmTokenizer(model)
    lines = _sample_lines(corpus, 4000)
    return {
        "model": str(model),
        "stats": _stats(pieces),
        "length_hist": _length_hist(pieces),
        "char_coverage": _char_coverage(pieces, corpus_chars),
        "affixes": _affix_hits(set(pieces)),
        "prefix_inventory": _prefix_inventory(pieces)[:80],
        "suffix_inventory": _continuation_inventory(pieces)[:80],
        "fertility": _fertility(tok, lines),
        "longest_pieces": sorted(
            (p for p in pieces if p not in _SPECIAL),
            key=lambda p: len(_body(p)),
            reverse=True,
        )[:20],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--speech-text",
        type=Path,
        default=ROOT / "exp" / "20m_bpe" / "corpus_1024.txt",
        help="Same corpus the speech unigram was trained on (train.jsonl dump).",
    )
    p.add_argument(
        "--speech-text-fallback",
        type=Path,
        default=ROOT / "exp" / "20m_char_4layer" / "lm" / "train.txt",
    )
    p.add_argument(
        "--word-text",
        type=Path,
        default=ROOT / "exp" / "20m_char_12layer" / "lm_word" / "train.txt",
    )
    p.add_argument(
        "--speech-unigram",
        type=Path,
        default=ROOT / "exp" / "20m_bpe" / "spm.model",
    )
    p.add_argument(
        "--word-unigram",
        type=Path,
        default=ROOT / "exp" / "tokenizers" / "unigram_1024_word.model",
    )
    p.add_argument("--out-dir", type=Path, default=ROOT / "exp" / "tokenizers")
    p.add_argument("--vocab-size", type=int, default=1024)
    args = p.parse_args()

    speech_text = args.speech_text if args.speech_text.exists() else args.speech_text_fallback
    if not speech_text.exists():
        log.error("missing speech corpus %s", speech_text)
        return 1
    if not args.word_text.exists():
        log.error("missing word corpus %s", args.word_text)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    speech_bpe = _ensure_model(
        speech_text,
        vocab_size=args.vocab_size,
        model_type="bpe",
        prefix=args.out_dir / f"bpe_{args.vocab_size}_speech",
    )
    word_bpe = _ensure_model(
        args.word_text,
        vocab_size=args.vocab_size,
        model_type="bpe",
        prefix=args.out_dir / f"bpe_{args.vocab_size}_word",
    )

    log.info("scanning Fidel charsets")
    speech_chars = _corpus_fidel(speech_text)
    word_chars = _corpus_fidel(args.word_text)

    models = {
        "speech_unigram": args.speech_unigram,
        "speech_bpe": speech_bpe,
        "word_unigram": args.word_unigram,
        "word_bpe": word_bpe,
    }
    for name, path in models.items():
        if not path.exists():
            log.error("missing %s %s", name, path)
            return 1

    profiles = {
        "speech_unigram": _profile(args.speech_unigram, speech_text, speech_chars),
        "speech_bpe": _profile(speech_bpe, speech_text, speech_chars),
        "word_unigram": _profile(args.word_unigram, args.word_text, word_chars),
        "word_bpe": _profile(word_bpe, args.word_text, word_chars),
    }

    def plist(path: Path) -> list[str]:
        return [p for p, _ in _pieces(path.with_suffix(".vocab"))]

    speech_bpe_pieces = plist(speech_bpe)
    word_bpe_pieces = plist(word_bpe)
    speech_uni_pieces = plist(args.speech_unigram)
    word_uni_pieces = plist(args.word_unigram)

    report = {
        "speech_text": str(speech_text),
        "word_text": str(args.word_text),
        "profiles": profiles,
        "overlaps": {
            "speech_unigram_vs_speech_bpe": _overlap(speech_uni_pieces, speech_bpe_pieces),
            "word_unigram_vs_word_bpe": _overlap(word_uni_pieces, word_bpe_pieces),
            "speech_bpe_vs_word_bpe": _overlap(speech_bpe_pieces, word_bpe_pieces),
            "speech_unigram_vs_word_unigram": _overlap(speech_uni_pieces, word_uni_pieces),
        },
        "shared_bpe_prefixes": sorted(
            set(_prefix_inventory(speech_bpe_pieces)) & set(_prefix_inventory(word_bpe_pieces)),
            key=lambda s: (-len(_body(s)), s),
        ),
        "shared_bpe_suffixes": sorted(
            set(_continuation_inventory(speech_bpe_pieces)) & set(_continuation_inventory(word_bpe_pieces)),
            key=lambda s: (-len(s), s),
        ),
    }

    out = args.out_dir / "compare_bpe_vs_unigram.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("wrote %s", out)

    def _line(name: str) -> None:
        pr = profiles[name]
        cov = pr["char_coverage"]
        aff = pr["affixes"]
        print(
            f"{name:16s}  atomic_fidel={cov['atomic_fidel_pieces']:3d}  "
            f"corpus_atomic={cov['pct_corpus_atomic']:5.1f}%  "
            f"buried={cov['buried_only']:3d}  missing={cov['missing_from_vocab']:3d}  "
            f"mean_len={pr['stats']['mean_piece_len']:.2f}  "
            f"wordlike={pr['stats']['pct_wordlike_underscore']:.1f}%  "
            f"ppw={pr['fertility']['pieces_per_word']:.3f}  "
            f"prefixes={aff['n_word_initial']}"
        )

    print("character coverage + fertility")
    for name in ("speech_unigram", "speech_bpe", "word_unigram", "word_bpe"):
        _line(name)
    print("overlaps", json.dumps(report["overlaps"], ensure_ascii=False))
    print("shared BPE prefixes:", " ".join(report["shared_bpe_prefixes"][:40]))
    print("shared BPE suffixes:", " ".join(report["shared_bpe_suffixes"][:40]))
    print("speech BPE longest:", " | ".join(profiles["speech_bpe"]["longest_pieces"][:12]))
    print("word   BPE longest:", " | ".join(profiles["word_bpe"]["longest_pieces"][:12]))
    print("speech BPE prefixes:", " ".join(profiles["speech_bpe"]["prefix_inventory"][:30]))
    print("word   BPE prefixes:", " ".join(profiles["word_bpe"]["prefix_inventory"][:30]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
