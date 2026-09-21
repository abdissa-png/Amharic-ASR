"""Train a SentencePiece unigram / BPE tokenizer on Dataset.ET train text.

Usage::

    python -m tokenizer.train_bpe --vocab-size 1024
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data.prepare import load_split_jsonl
from utils import ROOT, setup_logging

log = setup_logging()


def train_spm(
    texts: list[str],
    *,
    vocab_size: int,
    model_type: str,
    out_prefix: Path,
) -> Path:
    import sentencepiece as spm

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    corpus = out_prefix.parent / f"corpus_{vocab_size}.txt"
    corpus.write_text("\n".join(texts) + "\n", encoding="utf-8")
    # CTC blank is added by SpmTokenizer (id 0), so SentencePiece itself
    # does not reserve a blank piece. Request vocab_size - 1 pieces.
    spm.SentencePieceTrainer.Train(
        input=str(corpus),
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
        num_threads=4,
        minloglevel=1,
    )
    return Path(str(out_prefix) + ".model")


class SpmTokenizer:
    """SentencePiece wrapper with CTC blank at index 0 (SP ids shifted +1)."""

    def __init__(self, model_path: str | Path) -> None:
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self.model_path = Path(model_path)
        self._blank = 0
        self._offset = 1

    @property
    def blank_id(self) -> int:
        return self._blank

    @property
    def vocab_size(self) -> int:
        return int(self.sp.get_piece_size()) + self._offset

    @property
    def unk_id(self) -> int:
        return int(self.sp.unk_id()) + self._offset

    @property
    def tokens(self) -> list[str]:
        """CTC labels for pyctcdecode.

        Keep SentencePiece ``▁`` so the alphabet is detected as BPE. Replacing it
        with a space makes pyctcdecode treat pieces as characters; the word LM
        then almost never fires and WER gets worse than greedy.
        """
        labels = [""]
        for i in range(self.sp.get_piece_size()):
            piece = self.sp.id_to_piece(i)
            if piece == "<pad>":
                # "<pad>" would be rewritten to blank and duplicate id 0.
                labels.append("\u200b")
            else:
                labels.append(piece)
        return labels

    def encode(self, text: str) -> list[int]:
        return [i + self._offset for i in self.sp.encode(text, out_type=int)]

    def decode(self, ids, *, collapse: bool = False) -> str:
        ids = [int(i) for i in ids]
        if collapse:
            collapsed = []
            prev = None
            for i in ids:
                if i == self.blank_id or i == prev:
                    prev = i
                    continue
                prev = i
                collapsed.append(i)
            ids = collapsed
        sp_ids = [i - self._offset for i in ids if i != self.blank_id and i >= self._offset]
        return self.sp.decode(sp_ids)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--model-type", default="unigram", choices=["unigram", "bpe"])
    p.add_argument("--out-dir", type=Path, default=ROOT / "exp" / "tokenizers")
    args = p.parse_args()
    rows = load_split_jsonl("train")
    texts = [r["text"] for r in rows if r["text"]]
    prefix = args.out_dir / f"{args.model_type}_{args.vocab_size}"
    path = train_spm(texts, vocab_size=args.vocab_size, model_type=args.model_type, out_prefix=prefix)
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
