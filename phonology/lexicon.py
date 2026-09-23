"""Pronunciation lexicon: word → one or more phoneme sequences.

Never force a single pronunciation when G2P is ambiguous. The first entry is
the default (rule-based G2P); extra entries can be added from a TSV.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from phonology.g2p import g2p


class Lexicon:
    def __init__(self) -> None:
        self._prons: dict[str, list[list[str]]] = defaultdict(list)

    def add(self, word: str, pron: list[str]) -> None:
        if pron not in self._prons[word]:
            self._prons[word].append(pron)

    def get(self, word: str) -> list[list[str]]:
        if word in self._prons:
            return self._prons[word]
        return [g2p(word)]

    def words(self) -> list[str]:
        return sorted(self._prons)

    def __len__(self) -> int:
        return len(self._prons)

    def to_tsv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for word, prons in sorted(self._prons.items()):
                for pron in prons:
                    fh.write(f"{word}\t{' '.join(pron)}\n")

    @classmethod
    def from_words(cls, words: list[str]) -> "Lexicon":
        lex = cls()
        for w in words:
            lex.add(w, g2p(w))
        return lex

    @classmethod
    def from_tsv(cls, path: str | Path) -> "Lexicon":
        lex = cls()
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                word, *rest = line.split("\t")
                pron = rest[0].split() if rest else g2p(word)
                lex.add(word, pron)
        return lex
