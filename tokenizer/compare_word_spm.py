"""Train a word corpus unigram SentencePiece model and compare it to the speech one.

Uses the already-cleaned KenLM corpus (sentence-split, decontaminated)
so the word corpus pieces see the same text the word LM saw.

    python -m tokenizer.compare_word_spm
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from tokenizer.train_bpe import SpmTokenizer
from utils import ROOT, setup_logging

log = setup_logging()

_ETH = re.compile(r"[\u1200-\u137F]")
_SPECIAL = frozenset({"<unk>", "<pad>", "▁"})


def _pieces(vocab_path: Path) -> list[tuple[str, float]]:
    rows = []
    for line in vocab_path.read_text(encoding="utf-8").splitlines():
        piece, _, score = line.partition("\t")
        if not piece:
            continue
        try:
            rows.append((piece, float(score)))
        except ValueError:
            rows.append((piece, 0.0))
    return rows


def _stats(pieces: list[str]) -> dict:
    real = [p for p in pieces if p not in _SPECIAL]
    n = max(1, len(real))
    one = sum(1 for p in real if len(p) == 1 or (p.startswith("▁") and len(p) == 2))
    fidel1 = 0
    for p in real:
        body = p[1:] if p.startswith("▁") else p
        if len(body) == 1 and _ETH.search(body):
            fidel1 += 1
    latin = sum(1 for p in real if re.search(r"[A-Za-z0-9]", p))
    wordlike = sum(1 for p in real if p.startswith("▁") and len(p) > 2)
    lengths = [len(p[1:] if p.startswith("▁") else p) or 1 for p in real]
    return {
        "n_pieces": len(pieces),
        "n_content": len(real),
        "pct_one_symbol": round(100.0 * one / n, 1),
        "pct_one_fidel": round(100.0 * fidel1 / n, 1),
        "pct_latin_digit": round(100.0 * latin / n, 1),
        "pct_wordlike_underscore": round(100.0 * wordlike / n, 1),
        "mean_piece_len": round(sum(lengths) / n, 2),
        "max_piece_len": max(lengths) if lengths else 0,
    }


def _fertility(tok: SpmTokenizer, lines: list[str]) -> dict:
    n_words = n_pieces = 0
    whole = 0
    for line in lines:
        words = line.split()
        n_words += len(words)
        n_pieces += len(tok.encode(line))
        for w in words:
            ids = tok.encode(w)
            if len(ids) == 1:
                whole += 1
    return {
        "n_words": n_words,
        "n_pieces": n_pieces,
        "pieces_per_word": round(n_pieces / max(1, n_words), 3),
        "pct_whole_word_pieces": round(100.0 * whole / max(1, n_words), 1),
    }


def _sample_lines(path: Path, n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    reservoir: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            text = line.strip()
            if not text:
                continue
            if len(reservoir) < n:
                reservoir.append(text)
            else:
                j = rng.randint(0, i)
                if j < n:
                    reservoir[j] = text
    return reservoir


def train_from_file(text_path: Path, *, vocab_size: int, model_type: str, out_prefix: Path) -> Path:
    import sentencepiece as spm

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict = dict(
        input=str(text_path),
        model_prefix=str(out_prefix),
        vocab_size=max(8, vocab_size - 1),
        model_type=model_type,
        character_coverage=1.0,
        unk_id=0,
        bos_id=-1,
        eos_id=-1,
        pad_id=1,
        unk_piece="<unk>",
        pad_piece="<pad>",
        num_threads=8,
        minloglevel=1,
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
    )
    # seed_sentencepiece_size is unigram-only; BPE grows from characters via merges.
    if model_type == "unigram":
        kwargs["seed_sentencepiece_size"] = 1_000_000
    spm.SentencePieceTrainer.Train(**kwargs)
    return Path(str(out_prefix) + ".model")


def compare(speech_model: Path, word_model: Path, speech_txt: Path, word_txt: Path) -> dict:
    speech_vocab = speech_model.with_suffix(".vocab")
    word_vocab = word_model.with_suffix(".vocab")
    sp = [p for p, _ in _pieces(speech_vocab)]
    wp = [p for p, _ in _pieces(word_vocab)]
    ss, ws = set(sp), set(wp)
    both = ss & ws
    report = {
        "speech_model": str(speech_model),
        "word_model": str(word_model),
        "overlap": {
            "shared": len(both),
            "speech_only": len(ss - ws),
            "word_only": len(ws - ss),
            "jaccard": round(len(both) / max(1, len(ss | ws)), 3),
        },
        "speech_vocab_stats": _stats(sp),
        "word_vocab_stats": _stats(wp),
        "speech_only_examples": sorted(ss - ws, key=len, reverse=True)[:25],
        "word_only_examples": sorted(ws - ss, key=len, reverse=True)[:25],
        "shared_examples": sorted(both, key=len, reverse=True)[:20],
    }
    speech_tok, word_tok = SpmTokenizer(speech_model), SpmTokenizer(word_model)
    speech_lines = _sample_lines(speech_txt, 4000)
    word_lines = _sample_lines(word_txt, 4000)
    report["fertility_on_speech_text"] = {
        "speech_spm": _fertility(speech_tok, speech_lines),
        "word_spm": _fertility(word_tok, speech_lines),
    }
    report["fertility_on_word_text"] = {
        "speech_spm": _fertility(speech_tok, word_lines),
        "word_spm": _fertility(word_tok, word_lines),
    }
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--speech-model", type=Path, default=ROOT / "exp" / "20m_bpe" / "spm.model")
    p.add_argument(
        "--word-text",
        type=Path,
        default=ROOT / "exp" / "20m_char_12layer" / "lm_word" / "train.txt",
        help="Cleaned word corpus lines (KenLM train.txt), not raw word corpus.txt",
    )
    p.add_argument(
        "--speech-text",
        type=Path,
        default=ROOT / "exp" / "20m_char_4layer" / "lm" / "train.txt",
    )
    p.add_argument("--out-dir", type=Path, default=ROOT / "exp" / "tokenizers")
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--model-type", default="unigram", choices=["unigram", "bpe"])
    args = p.parse_args()

    if not args.speech_model.exists():
        log.error("missing speech SPM %s", args.speech_model)
        return 1
    if not args.word_text.exists():
        log.error("missing word corpus %s — train KenLM first or pass --word-text", args.word_text)
        return 1

    prefix = args.out_dir / f"{args.model_type}_{args.vocab_size}_word"
    model_path = Path(str(prefix) + ".model")
    if model_path.exists():
        log.info("reusing %s", model_path)
    else:
        log.info("training %s on %s", args.model_type, args.word_text)
        model_path = train_from_file(
            args.word_text,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            out_prefix=prefix,
        )
        log.info("wrote %s", model_path)

    report = compare(args.speech_model, model_path, args.speech_text, args.word_text)
    out = args.out_dir / "compare_speech_vs_word.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("wrote %s", out)
    ov = report["overlap"]
    print(
        f"overlap  shared={ov['shared']}  speech_only={ov['speech_only']}  "
        f"word_only={ov['word_only']}  jaccard={ov['jaccard']}"
    )
    print("speech vocab", json.dumps(report["speech_vocab_stats"], ensure_ascii=False))
    print("word   vocab", json.dumps(report["word_vocab_stats"], ensure_ascii=False))
    print("fertility on speech transcripts", json.dumps(report["fertility_on_speech_text"], ensure_ascii=False))
    print("fertility on word sample     ", json.dumps(report["fertility_on_word_text"], ensure_ascii=False))
    print("word-only (longest):", " | ".join(report["word_only_examples"][:12]))
    print("speech-only (longest):", " | ".join(report["speech_only_examples"][:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
