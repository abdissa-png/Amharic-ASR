"""Light-weight Amharic affix stripping used as *lexical structure*, not as a stemmer.

The goal is to ask: can an unseen surface form be explained as a known stem
plus a known affix? That classification feeds OOV analysis and, later, a
decomposition-aware lexicon (Phase 10).
"""

from __future__ import annotations

# Frequent bound morphemes. Order: longest first so we peel the largest match.
PREFIXES = (
    "እንደምን",
    "እንዲህ",
    "እስከ",
    "እንደ",
    "ስለ",
    "ያለ",
    "ከመ",
    "አስተ",
    "አስ",
    "ተ",
    "እን",
    "የ",
    "ለ",
    "በ",
    "ከ",
    "ስ",
    "አል",
    "አት",
    "አይ",
)

SUFFIXES = (
    "አቸውን",
    "አቸው",
    "ዎቻቸው",
    "ዎቹ",
    "ዎች",
    "ወች",
    "ኦች",
    "ችው",
    "ችሁ",
    "ለት",
    "ለች",
    "ለው",
    "ቸው",
    "አት",
    "ዋለ",
    "ዋል",
    "ነው",
    "ናት",
    "ናቸው",
    "ንም",
    "ውም",
    "ዋም",
    "ች",
    "ው",
    "ዋ",
    "ን",
    "ም",
    "ት",
    "ህ",
    "ሽ",
    "ኝ",
    "ኩ",
    "ክ",
    "ሁ",
    "ይ",
    "ዎ",
)

MIN_STEM = 2

# Ethiopic 1st-order bases used to rewrite the last syllable's vowel order
# (definite -u, feminine -wa, etc. are not separate suffix letters).
_FIDEL_BASES = (
    0x1200, 0x1208, 0x1210, 0x1218, 0x1220, 0x1228, 0x1230, 0x1238,
    0x1240, 0x1260, 0x1268, 0x1270, 0x1278, 0x1280, 0x1290, 0x1298,
    0x12A0, 0x12A8, 0x12B8, 0x12C8, 0x12D0, 0x12D8, 0x12E0, 0x12E8,
    0x12F0, 0x1300, 0x1308, 0x1320, 0x1328, 0x1330, 0x1338, 0x1340,
    0x1348, 0x1350,
)


def _rewrite_last_order(word: str) -> list[str]:
    """Peel fused vowel-order morphology on the last Fidel (e.g. ቤቱ → ቤት)."""
    if not word:
        return []
    cp = ord(word[-1])
    for base in _FIDEL_BASES:
        if base <= cp < base + 8:
            order = cp - base
            alts = []
            for o in (5, 0):  # 6th-order bare stem, 1st-order ä stem
                if o != order:
                    alts.append(word[:-1] + chr(base + o))
            return alts
    return []


def strip_once(word: str, affixes: tuple[str, ...], *, prefix: bool) -> str | None:
    for aff in affixes:
        if prefix and word.startswith(aff) and len(word) - len(aff) >= MIN_STEM:
            return word[len(aff) :]
        if not prefix and word.endswith(aff) and len(word) - len(aff) >= MIN_STEM:
            return word[: -len(aff)]
    return None


def candidate_stems(word: str, *, max_peels: int = 3) -> set[str]:
    """Generate stripped variants by peeling prefixes/suffixes and vowel orders."""
    seen = {word}
    frontier = {word}
    for _ in range(max_peels):
        nxt: set[str] = set()
        for w in frontier:
            p = strip_once(w, PREFIXES, prefix=True)
            s = strip_once(w, SUFFIXES, prefix=False)
            if p:
                nxt.add(p)
            if s:
                nxt.add(s)
            nxt.update(_rewrite_last_order(w))
        nxt -= seen
        if not nxt:
            break
        seen |= nxt
        frontier = nxt
    return seen


def likely_morphological_variant(word: str, vocab: set[str]) -> bool:
    if word in vocab:
        return False
    return any(stem in vocab for stem in candidate_stems(word) if stem != word)


def split_compound(word: str, vocab: set[str], *, min_part: int = 2) -> tuple[str, str] | None:
    if word in vocab or len(word) < 2 * min_part:
        return None
    for i in range(min_part, len(word) - min_part + 1):
        a, b = word[:i], word[i:]
        if a in vocab and b in vocab:
            return a, b
    return None


def surface_morphs(word: str) -> list[str]:
    """Greedy longest-match prefix then suffix peel. The remainder is the stem.

    Decode-time segmentation for a morpheme n-gram: CTC still emits characters,
    and the LM is scored when a word boundary (space) closes.
    """
    if not word:
        return []
    prefixes: list[str] = []
    w = word
    while True:
        hit = None
        for aff in PREFIXES:
            if w.startswith(aff) and len(w) - len(aff) >= MIN_STEM:
                hit = aff
                break
        if hit is None:
            break
        prefixes.append(hit)
        w = w[len(hit) :]
    suffixes: list[str] = []
    while True:
        hit = None
        for aff in SUFFIXES:
            if w.endswith(aff) and len(w) - len(aff) >= MIN_STEM:
                hit = aff
                break
        if hit is None:
            break
        suffixes.append(hit)
        w = w[: -len(hit)]
    stem = [w] if w else []
    return prefixes + stem + list(reversed(suffixes))


def decompose(word: str, vocab: set[str]) -> dict[str, object]:
    stems = sorted(candidate_stems(word), key=len)
    known = [s for s in stems if s in vocab]
    compound = split_compound(word, vocab)
    return {
        "word": word,
        "stems": stems,
        "known_stems": known,
        "compound": list(compound) if compound else None,
        "morphological_variant": likely_morphological_variant(word, vocab),
    }
