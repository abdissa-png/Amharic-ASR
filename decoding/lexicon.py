"""Lexicon-constrained decoding helpers (Decoders D–F).

The first custom systems use greedy/beam CTC. Lexicon constraints are wired
here so G2P / morphological experiments can plug in without rewriting the
acoustic model.
"""

from __future__ import annotations

from phonology.lexicon import Lexicon
from phonology.decomposition import decompose


def lexicon_from_train_words(words: list[str]) -> Lexicon:
    return Lexicon.from_words(sorted(set(words)))


def decomposition_lexicon(words: list[str], vocab: set[str]) -> dict[str, dict]:
    return {w: decompose(w, vocab) for w in words}
