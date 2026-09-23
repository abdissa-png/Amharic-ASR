"""Rule-based Amharic G2P from Fidel (Ge'ez syllabary).

Each Fidel letter is a CV syllable: a consonant base plus a vowel order.
Labiovelar extras (ቀ family, etc.) are mapped as CwV. This is phonological
structure, not a claim of unique pronunciation — the lexicon (``lexicon.py``)
is allowed to store multiple pronunciations per word.
"""

from __future__ import annotations

import re
from functools import lru_cache

# Vowel orders 1–7 (ä u i a e ɨ o). Order 8 is typically a labialised /wä/-like
# form and is handled per-family below.
VOWELS = ("ä", "u", "i", "a", "e", "ɨ", "o")

# Unicode base of the 1st-order (ä) glyph → IPA-like consonant.
# Sources: standard Fidel chart; homophones kept distinct (training-side).
BASES: dict[int, str] = {
    0x1200: "h",   # ሀ
    0x1208: "l",   # ለ
    0x1210: "h",   # ሐ  (ħ ~ h in modern Amharic)
    0x1218: "m",   # መ
    0x1220: "s",   # ሠ
    0x1228: "r",   # ረ
    0x1230: "s",   # ሰ
    0x1238: "ʃ",   # ሸ
    0x1240: "k'",  # ቀ
    0x1260: "b",   # በ
    0x1268: "v",   # ቨ
    0x1270: "t",   # ተ
    0x1278: "tʃ",  # ቸ
    0x1280: "h",   # ኀ
    0x1290: "n",   # ነ
    0x1298: "ɲ",   # ኘ
    0x12A0: "ʔ",   # አ
    0x12A8: "k",   # ከ
    0x12B8: "x",   # ኸ
    0x12C8: "w",   # ወ
    0x12D0: "ʕ",   # ዐ
    0x12D8: "z",   # ዘ
    0x12E0: "ʒ",   # ዠ
    0x12E8: "j",   # የ
    0x12F0: "d",   # ደ
    0x1300: "dʒ",  # ጀ
    0x1308: "g",   # ገ
    0x1320: "t'",  # ጠ
    0x1328: "tʃ'", # ጨ
    0x1330: "p'",  # ጰ
    0x1338: "ts'", # ጸ
    0x1340: "ts'", # ፀ
    0x1348: "f",   # ፈ
    0x1350: "p",   # ፐ
}

# Labiovelar families: 1st-order codepoint → (consonant, vowels for the series)
# ቈ 0x1248, ኰ 0x12B0, ጐ 0x1310, ኈ 0x1288  — typically Cʷä Cʷi Cʷa Cʷe Cʷe/Cʷɨ
LABIOVELAR = {
    0x1248: "k'w",
    0x1288: "hw",
    0x12B0: "kw",
    0x1310: "gw",
}

_SPACE = re.compile(r"\s+")


def _syllable(consonant: str, vowel: str) -> str:
    if consonant in {"ʔ", "ʕ"}:
        return vowel
    if vowel == "ɨ":
        return consonant  # 6th order is often the bare consonant
    return consonant + vowel


@lru_cache(maxsize=4096)
def fidel_to_phonemes(ch: str) -> str | None:
    cp = ord(ch)
    if ch.isspace():
        return " "
    if "0" <= ch <= "9":
        return ch
    if ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
        return ch.lower()

    for base, cons in BASES.items():
        if base <= cp < base + 8:
            return _syllable(cons, VOWELS[cp - base])

    for base, cons in LABIOVELAR.items():
        # series of 5–6 labialised orders
        if base <= cp <= base + 7:
            idx = cp - base
            # common layout: ä, unused/u, i, a, e, ɨ, o-like
            vowels = ("ä", "u", "i", "a", "e", "ɨ", "o", "ä")
            return _syllable(cons, vowels[min(idx, 7)])
    return None


def g2p(word: str) -> list[str]:
    """Phoneme tokens for one whitespace-delimited word."""
    phones: list[str] = []
    for ch in word:
        p = fidel_to_phonemes(ch)
        if p is None:
            continue
        if p == " ":
            continue
        phones.append(p)
    return phones


def g2p_utterance(text: str) -> list[str]:
    """Phoneme sequence for an utterance, with word-boundary token ``|``."""
    words = _SPACE.split(text.strip())
    seq: list[str] = []
    for i, w in enumerate(words):
        seq.extend(g2p(w))
        if i != len(words) - 1:
            seq.append("|")
    return seq


def phoneme_inventory() -> list[str]:
    inv = set()
    for cons in list(BASES.values()) + list(LABIOVELAR.values()):
        for v in VOWELS:
            inv.add(_syllable(cons, v))
    inv.update(str(d) for d in range(10))
    inv.add("|")
    return sorted(inv)
