"""Longest-match subword tokenizer for a filtered CTC piece inventory.

SentencePiece BPE encoding follows a merge table, so dropping word corpus-unstable
pieces requires a different encoder. This class greedy-matches the remaining
set (WordPiece-style) and keeps ``▁`` so KenLM still treats the alphabet as BPE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from tokenizer.char import BLANK, PAD, UNK

MARK = "\u2581"


class PieceTokenizer:
    def __init__(self, tokens: list[str]) -> None:
        if tokens[0] != BLANK:
            raise ValueError("blank must be index 0 for CTC")
        if UNK not in tokens or PAD not in tokens:
            raise ValueError("vocab must include <unk> and <pad>")
        self._tokens = list(tokens)
        # KenLM / pyctcdecode: keep ▁ pieces, never emit "<pad>" as a label.
        self.tokens = ["\u200b" if t == PAD else t for t in self._tokens]
        self.stoi = {t: i for i, t in enumerate(self._tokens)}
        self.itos = self._tokens
        self._max_len = max((len(t) for t in self._tokens if t not in (BLANK, UNK, PAD)), default=1)

    @property
    def blank_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for word in text.split():
            ids.extend(self._encode_word(word))
        return ids

    def _encode_word(self, word: str) -> list[int]:
        s = MARK + word
        ids: list[int] = []
        i = 0
        n = len(s)
        while i < n:
            end = min(n, i + self._max_len)
            matched = None
            while end > i:
                piece = s[i:end]
                if piece in self.stoi:
                    matched = piece
                    break
                end -= 1
            if matched is None:
                ids.append(self.stoi.get(s[i], self.unk_id))
                i += 1
            else:
                ids.append(self.stoi[matched])
                i += len(matched)
        return ids

    def decode(self, ids: Iterable[int], *, collapse: bool = False) -> str:
        pieces: list[str] = []
        prev = None
        for i in ids:
            i = int(i)
            if collapse:
                if i == self.blank_id or i == prev:
                    prev = i
                    continue
                prev = i
            if i == self.blank_id:
                continue
            tok = self.itos[i] if 0 <= i < len(self.itos) else UNK
            if tok in (UNK, PAD, BLANK, "\u200b"):
                continue
            pieces.append(tok)
        out: list[str] = []
        for p in pieces:
            if p.startswith(MARK):
                if out:
                    out.append(" ")
                out.append(p[len(MARK) :])
            else:
                out.append(p)
        return "".join(out)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"kind": "piece", "tokens": self._tokens}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "PieceTokenizer":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(obj["tokens"])
