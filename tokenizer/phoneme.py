"""Phoneme-sequence tokenizer for CTC (Phase 9)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from phonology.g2p import g2p_utterance, phoneme_inventory
from tokenizer.char import BLANK, PAD, UNK


class PhonemeTokenizer:
    def __init__(self, tokens: list[str]) -> None:
        if tokens[0] != BLANK:
            raise ValueError("blank must be index 0")
        self.tokens = tokens
        self.stoi = {t: i for i, t in enumerate(tokens)}

    @property
    def blank_id(self) -> int:
        return 0

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        ids = []
        for p in g2p_utterance(text):
            ids.append(self.stoi.get(p, self.stoi[UNK]))
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
            tok = self.tokens[i] if 0 <= i < len(self.tokens) else UNK
            if tok in (UNK, PAD, BLANK):
                continue
            if tok == "|":
                pieces.append(" ")
            else:
                pieces.append(tok)
        return "".join(pieces).strip()

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"tokens": self.tokens}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PhonemeTokenizer":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(obj["tokens"])

    @classmethod
    def default(cls) -> "PhonemeTokenizer":
        inv = phoneme_inventory()
        # Also allow raw Latin letters that G2P may emit.
        extra = [chr(c) for c in range(ord("a"), ord("z") + 1)]
        tokens = [BLANK, UNK, PAD] + sorted(set(inv) | set(extra))
        return cls(tokens)
