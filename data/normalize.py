"""Amharic transcript normalisation.

Two policies are kept strictly separate:

* ``for_training`` — conservative. NFC, whitespace, quote unification.
  Homophones are **not** folded. Punctuation is mapped to space so CTC
  does not spend capacity on marks that evaluation strips anyway.
* ``for_eval`` — Dataset.ET official scoring (see ``evaluation.geez_eval``):
  NFC, Ge'ez homophone folding, strip punctuation/symbols, casefold.

Do not train on the eval policy. Folding ሐ into ሀ is correct when comparing
two strings and wrong as a training target (Dataset.ET METHOD.md).
"""

from __future__ import annotations

import re
import unicodedata

from evaluation.geez_eval import normalize as eval_normalize

_WS = re.compile(r"\s+")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Unify Arabic / Latin / typographic punctuation to a small ASCII set.
_QUOTE_MAP = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "«": '"',
        "»": '"',
        "‹": "'",
        "›": "'",
        "‘": "'",
        "’": "'",
        "‚": "'",
        "´": "'",
        "`": "'",
        "–": "-",
        "—": "-",
        "−": "-",
        "…": "...",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "；": ";",
        "：": ":",
        "፡": " ",  # Ethiopic wordspace → space (not a letter)
    }
)

# Ethiopic punctuation block members commonly appearing in prompts.
_ETHIOPIC_PUNCT = "፠፡።፣፤፥፦፧፨"

# Latin/Arabic punctuation we treat as separators for the ASR training target.
_LATIN_PUNCT = r"""!"#$%&()*+,./:;<=>?@[\]^_{|}~"""


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def collapse_whitespace(text: str) -> str:
    return _WS.sub(" ", text).strip()


def unify_punctuation(text: str) -> str:
    return text.translate(_QUOTE_MAP)


def strip_control(text: str) -> str:
    return _CTRL.sub("", text)


def punctuation_to_space(text: str) -> str:
    """Replace punctuation/symbols with space; keep letters, marks, digits."""
    out: list[str] = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] in ("P", "S") or ch in _ETHIOPIC_PUNCT:
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def for_training(text: str) -> str:
    """Conservative target used as CTC supervision.

    Keeps Ge'ez homophone distinctions. Drops punctuation (eval strips it).
    Leaves digits and English tokens intact. Does not collapse repeated
    letters (gemination / emphasis may be real).
    """
    text = nfc(text)
    text = strip_control(text)
    text = unify_punctuation(text)
    text = punctuation_to_space(text)
    return collapse_whitespace(text)


def for_eval(text: str) -> str:
    """Dataset.ET scoring normalisation. Comparable with published figures."""
    return eval_normalize(text, fold_geez=True)


def tokenize_words(text: str, *, policy: str = "eval") -> list[str]:
    if policy == "eval":
        normed = for_eval(text)
    elif policy == "train":
        normed = for_training(text)
    else:
        raise ValueError(f"unknown policy {policy}")
    return [w for w in normed.split() if w]


def charset(text: str) -> set[str]:
    return set(for_training(text))
