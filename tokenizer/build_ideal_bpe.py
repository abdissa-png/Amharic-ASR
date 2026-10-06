"""Build a constrained Amharic CTC piece vocab.

Locks speech characters and productive affixes, trains length-capped BPE on
speech, then keeps a merge only if it also fires on word corpus. Closed-class function
words that appear in both corpora are added last.

    python -m tokenizer.build_ideal_bpe
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from tokenizer.char import BLANK, PAD, UNK
from tokenizer.compare_word_spm import _ETH, _fertility, _sample_lines
from tokenizer.piece import MARK, PieceTokenizer
from tokenizer.train_bpe import SpmTokenizer
from utils import ROOT, setup_logging

log = setup_logging()

WORD_INITIAL = [
    f"{MARK}የ",
    f"{MARK}በ",
    f"{MARK}ለ",
    f"{MARK}ከ",
    f"{MARK}ወደ",
    f"{MARK}ስለ",
    f"{MARK}እንደ",
    f"{MARK}እስከ",
    f"{MARK}ያለ",
    f"{MARK}አል",
    f"{MARK}አይ",
    f"{MARK}ይ",
    f"{MARK}ት",
    f"{MARK}ን",
    f"{MARK}መ",
    f"{MARK}ተ",
    f"{MARK}አስ",
    f"{MARK}እየ",
    f"{MARK}የሚ",
    f"{MARK}የተ",
    f"{MARK}በመ",
    f"{MARK}ለማ",
    f"{MARK}እንዲ",
    f"{MARK}እንደሚ",
    f"{MARK}እንዳ",
]
CONTINUATION = ["ዎች", "ነት", "ዋል", "ናል", "ቸው", "ችው", "ውን", "ችን", "ቸውን"]
FUNCTION_WORDS = [
    "እና",
    "ነው",
    "ላይ",
    "ውስጥ",
    "ጋር",
    "ወደ",
    "ደግሞ",
    "ጊዜ",
    "እንደ",
    "ስለ",
    "እስከ",
    "ያለ",
    "ግን",
    "ወይም",
    "ሁሉ",
    "ሁሉም",
    "አሁን",
    "ብቻ",
    "ይህ",
    "እነዚህ",
    "ናቸው",
    "ሆኖ",
    "ሆነ",
    "እንጂ",
    "ዋና",
    "እዚህ",
    "እንዲሁም",
]


def _body(piece: str) -> str:
    return piece[len(MARK) :] if piece.startswith(MARK) else piece


def _has_fidel(text: str) -> bool:
    return bool(_ETH.search(text))


def _word_counts(path: Path, *, max_lines: int | None = None) -> Counter[str]:
    counts: Counter[str] = Counter()
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            counts.update(line.split())
            if max_lines is not None and i + 1 >= max_lines:
                break
    return counts


def _affix_support(piece: str, words: Counter[str]) -> int:
    body = _body(piece)
    if not body:
        return 0
    if piece.startswith(MARK):
        return sum(c for w, c in words.items() if w.startswith(body))
    return sum(c for w, c in words.items() if w.endswith(body) or body in w)


def _emit_counts(tok: SpmTokenizer, path: Path, *, max_lines: int) -> Counter[str]:
    counts: Counter[str] = Counter()
    sp = tok.sp
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            text = line.strip()
            if not text:
                continue
            for piece in sp.encode(text, out_type=str):
                counts[piece] += 1
            if i + 1 >= max_lines:
                break
    return counts


def _single_chars(text: str) -> tuple[list[str], list[str]]:
    fidel = sorted({c for c in text if _ETH.search(c)})
    other = sorted({c for c in text if c not in fidel and c != " " and c != "\n" and c != "\t"})
    return fidel, other


def _train_candidate(
    speech_text: Path,
    *,
    out_prefix: Path,
    vocab_size: int,
    max_piece_len: int,
    affixes: list[str],
) -> Path:
    import sentencepiece as spm

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict = dict(
        input=str(speech_text),
        model_prefix=str(out_prefix),
        vocab_size=max(64, vocab_size),
        model_type="bpe",
        character_coverage=1.0,
        unk_id=0,
        bos_id=-1,
        eos_id=-1,
        pad_id=1,
        unk_piece="<unk>",
        pad_piece="<pad>",
        num_threads=8,
        minloglevel=1,
        max_sentencepiece_length=max_piece_len,
        shuffle_input_sentence=True,
        input_sentence_size=2_000_000,
    )
    if affixes:
        kwargs["user_defined_symbols"] = ",".join(affixes)
    spm.SentencePieceTrainer.Train(**kwargs)
    return Path(str(out_prefix) + ".model")


def build(
    *,
    speech_text: Path,
    word_text: Path,
    out_dir: Path,
    final_size: int,
    candidate_size: int,
    max_piece_len: int,
    min_speech_affix: int,
    min_speech_merge: int,
    min_word_merge: int,
    min_speech_func: int,
    min_word_func: int,
    word_lines: int,
    extra_affixes: list[str] | None = None,
    bpe_text: Path | None = None,
) -> dict:
    speech_raw = speech_text.read_text(encoding="utf-8")
    speech_words = Counter(speech_raw.split())
    log.info("counting word corpus words (%s lines cap)", f"{word_lines:,}")
    word_words = _word_counts(word_text, max_lines=word_lines)

    fidel, other_chars = _single_chars(speech_raw)
    locked_affixes = []
    dropped_affixes = []
    for aff in WORD_INITIAL + CONTINUATION:
        n = _affix_support(aff, speech_words)
        if n >= min_speech_affix:
            locked_affixes.append(aff)
        else:
            dropped_affixes.append({"piece": aff, "speech_support": n})
    for aff in extra_affixes or []:
        if aff and aff not in locked_affixes:
            locked_affixes.append(aff)

    func_keep = []
    for w in FUNCTION_WORDS:
        sc, gc = speech_words[w], word_words[w]
        if sc >= min_speech_func and gc >= min_word_func:
            func_keep.append(f"{MARK}{w}")

    bpe_source = bpe_text or speech_text
    log.info("training length-capped BPE on %s (candidate=%d max_len=%d)", bpe_source, candidate_size, max_piece_len)
    cand_prefix = out_dir / f"candidate_bpe_{candidate_size}"
    cand_model = Path(str(cand_prefix) + ".model")
    if not cand_model.exists():
        spm_affixes = [a for a in locked_affixes if len(a) <= max_piece_len]
        cand_model = _train_candidate(
            bpe_source,
            out_prefix=cand_prefix,
            vocab_size=candidate_size,
            max_piece_len=max_piece_len,
            affixes=spm_affixes,
        )
    else:
        log.info("reusing %s", cand_model)

    cand = SpmTokenizer(cand_model)
    log.info("counting candidate piece emissions on speech")
    speech_emit = _emit_counts(cand, speech_text, max_lines=10_000_000)
    log.info("counting candidate piece emissions on word corpus")
    word_emit = _emit_counts(cand, word_text, max_lines=word_lines)

    required = [BLANK, UNK, PAD, MARK]
    for ch in fidel + other_chars:
        if ch not in required:
            required.append(ch)
    for aff in locked_affixes:
        if aff not in required:
            required.append(aff)
    for fw in func_keep:
        if fw not in required:
            required.append(fw)

    merges = []
    rejected = []
    for i in range(cand.sp.get_piece_size()):
        piece = cand.sp.id_to_piece(i)
        if piece in required or piece in ("<unk>", "<pad>", MARK):
            continue
        body = _body(piece)
        if not body or piece in (BLANK, UNK, PAD):
            continue
        sc, gc = speech_emit.get(piece, 0), word_emit.get(piece, 0)
        if len(body) < 2 or len(body) > max_piece_len:
            rejected.append({"piece": piece, "reason": "length", "speech": sc, "word": gc})
            continue
        if not _has_fidel(body):
            rejected.append({"piece": piece, "reason": "no_fidel", "speech": sc, "word": gc})
            continue
        if sc < min_speech_merge or gc < min_word_merge:
            rejected.append({"piece": piece, "reason": "support", "speech": sc, "word": gc})
            continue
        merges.append((piece, sc, gc))
    merges.sort(key=lambda row: (-row[1], -row[2], row[0]))
    merge_meta = {piece: {"piece": piece, "speech": sc, "word": gc} for piece, sc, gc in merges}

    tokens = list(required)
    seen = set(tokens)
    for piece, sc, gc in merges:
        if len(tokens) >= final_size:
            break
        if piece in seen:
            continue
        tokens.append(piece)
        seen.add(piece)

    speech_lines = [ln.strip() for ln in speech_raw.splitlines() if ln.strip()]
    protected = set(required)
    banned: set[str] = set()
    tok = PieceTokenizer(tokens)
    for _ in range(6):
        used: set[int] = set()
        for line in speech_lines:
            used.update(tok.encode(line))
        unused = [
            tok.itos[i]
            for i in range(tok.vocab_size)
            if i not in used and tok.itos[i] not in (BLANK, UNK, PAD)
        ]
        drop = [p for p in unused if p not in protected]
        if not drop:
            break
        banned.update(drop)
        tokens = [t for t in tokens if t not in banned]
        seen = set(tokens)
        for piece, sc, gc in merges:
            if len(tokens) >= final_size:
                break
            if piece in seen or piece in banned:
                continue
            tokens.append(piece)
            seen.add(piece)
        tok = PieceTokenizer(tokens)

    kept_merges = [merge_meta[p] for p in tokens if p in merge_meta]
    layers = {
        "specials": 4,
        "fidel": len(fidel),
        "other_chars": len(other_chars),
        "affixes": len(locked_affixes),
        "function_words": len(func_keep),
        "merges": len(kept_merges),
        "vocab_size": tok.vocab_size,
    }
    word_sample = _sample_lines(word_text, 4000)
    fert_speech = _fertility(tok, speech_lines[:4000] if len(speech_lines) > 4000 else speech_lines)
    fert_word = _fertility(tok, word_sample)

    used = set()
    for line in speech_lines:
        used.update(tok.encode(line))
    unused = [tok.itos[i] for i in range(tok.vocab_size) if i not in used and tok.itos[i] not in (BLANK, UNK, PAD)]

    lengths = [len(_body(p)) or 1 for p in tokens if p not in (BLANK, UNK, PAD, MARK)]
    report = {
        "speech_text": str(speech_text),
        "word_text": str(word_text),
        "candidate_model": str(cand_model),
        "layers": layers,
        "locked_affixes": locked_affixes,
        "dropped_affixes": dropped_affixes,
        "function_words": func_keep,
        "n_rejected_merges": len(rejected),
        "kept_merges_head": kept_merges[:40],
        "unused_on_speech": unused,
        "pruned_longest_match_dead": sorted(banned),
        "mean_piece_len": round(sum(lengths) / max(1, len(lengths)), 2),
        "max_piece_len": max(lengths) if lengths else 0,
        "fertility_on_speech": fert_speech,
        "fertility_on_word_sample": fert_word,
    }
    return {"tokenizer": tok, "report": report, "rejected": rejected}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--speech-text",
        type=Path,
        default=ROOT / "exp" / "20m_bpe" / "corpus_1024.txt",
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
    p.add_argument("--out-dir", type=Path, default=ROOT / "exp" / "tokenizers" / "ideal_bpe")
    p.add_argument("--final-size", type=int, default=1024)
    p.add_argument("--candidate-size", type=int, default=2048)
    p.add_argument("--max-piece-len", type=int, default=4)
    p.add_argument("--min-speech-affix", type=int, default=10)
    p.add_argument("--min-speech-merge", type=int, default=20)
    p.add_argument("--min-word-merge", type=int, default=20)
    p.add_argument("--min-speech-func", type=int, default=50)
    p.add_argument("--min-word-func", type=int, default=20)
    p.add_argument("--word-lines", type=int, default=200_000)
    args = p.parse_args()

    speech_text = args.speech_text if args.speech_text.exists() else args.speech_text_fallback
    if not speech_text.exists():
        log.error("missing speech corpus")
        return 1
    if not args.word_text.exists():
        log.error("missing word corpus")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = build(
        speech_text=speech_text,
        word_text=args.word_text,
        out_dir=args.out_dir,
        final_size=args.final_size,
        candidate_size=args.candidate_size,
        max_piece_len=args.max_piece_len,
        min_speech_affix=args.min_speech_affix,
        min_speech_merge=args.min_speech_merge,
        min_word_merge=args.min_word_merge,
        min_speech_func=args.min_speech_func,
        min_word_func=args.min_word_func,
        word_lines=args.word_lines,
    )
    tok: PieceTokenizer = result["tokenizer"]
    report = result["report"]
    tok.save(args.out_dir / "tokenizer.json")
    vocab_lines = [f"{t}\t{i}" for i, t in enumerate(tok.itos)]
    (args.out_dir / "vocab.tsv").write_text("\n".join(vocab_lines) + "\n", encoding="utf-8")
    (args.out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "rejected_merges.json").write_text(
        json.dumps(result["rejected"][:500], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("wrote %s", args.out_dir / "tokenizer.json")
    print("layers", json.dumps(report["layers"], ensure_ascii=False))
    print("affixes", " ".join(report["locked_affixes"]))
    print("function words", " ".join(report["function_words"]))
    print("dropped affixes", json.dumps(report["dropped_affixes"], ensure_ascii=False))
    print("unused on speech", report["unused_on_speech"])
    print("mean_piece_len", report["mean_piece_len"], "max", report["max_piece_len"])
    print("fertility speech", json.dumps(report["fertility_on_speech"], ensure_ascii=False))
    print("fertility word  ", json.dumps(report["fertility_on_word_sample"], ensure_ascii=False))
    print("kept merges", " ".join(m["piece"] for m in report["kept_merges_head"][:25]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
