"""Character vocabulary for CTC."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

BLANK = "<blank>"
UNK = "<unk>"
PAD = "<pad>"


class CharTokenizer:
    def __init__(self, tokens: list[str]) -> None:
        if tokens[0] != BLANK:
            raise ValueError("blank must be index 0 for CTC")
        self.tokens = tokens
        self.stoi = {t: i for i, t in enumerate(tokens)}
        self.itos = tokens

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
        for ch in text:
            ids.append(self.stoi.get(ch, self.unk_id))
        return ids

    def decode(self, ids: Iterable[int], *, collapse: bool = False) -> str:
        pieces: list[str] = []
        prev = None
        for i in ids:
            i = int(i)
            if collapse:
                if i == self.blank_id or i == prev:
                    prev = i if i != self.blank_id else prev
                    if i == self.blank_id:
                        prev = self.blank_id
                    continue
                prev = i
            if i == self.blank_id:
                continue
            tok = self.itos[i] if 0 <= i < len(self.itos) else UNK
            if tok in (UNK, PAD):
                continue
            pieces.append(tok)
        return "".join(pieces)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"tokens": self.tokens}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(obj["tokens"])

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> "CharTokenizer":
        chars = set()
        for t in texts:
            chars.update(t)
        # Stable order: space, then Ethiopic, then the rest.
        ordered = sorted(chars, key=lambda c: (0 if c == " " else 1 if "\u1200" <= c <= "\u137F" else 2, c))
        tokens = [BLANK, UNK, PAD] + ordered
        return cls(tokens)
