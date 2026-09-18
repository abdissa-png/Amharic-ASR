"""Character error rate wrapper (Dataset.ET scoring)."""

from __future__ import annotations

from typing import Sequence

from evaluation.metrics import score_texts


def cer(refs: Sequence[str], hyps: Sequence[str], *, fold_geez: bool = True) -> float:
    return score_texts(refs, hyps, fold_geez=fold_geez)["cer"]
