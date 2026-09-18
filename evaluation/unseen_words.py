"""Seen-word vs unseen-word WER from a word-level alignment."""

from __future__ import annotations

from typing import Sequence

from evaluation.geez_eval import normalize as geez_normalize
from evaluation.wer import alignment_ops


def seen_unseen_wer(
    refs: Sequence[str],
    hyps: Sequence[str],
    v_train: set[str],
    *,
    fold_geez: bool = True,
) -> dict[str, float]:
    """Error rates restricted to reference words that are seen / unseen in train.

    Insertions are charged to the overall rate only; they are not attributed
    to seen or unseen reference words.
    """
    seen_err = seen_n = 0
    unseen_err = unseen_n = 0
    for ref, hyp in zip(refs, hyps, strict=True):
        r = geez_normalize(ref, fold_geez=fold_geez)
        h = geez_normalize(hyp, fold_geez=fold_geez)
        if not r.strip():
            continue
        for op, rw, hw in alignment_ops(r, h):
            if rw is None:
                continue
            if rw in v_train:
                seen_n += 1
                if op != "equal":
                    seen_err += 1
            else:
                unseen_n += 1
                if op != "equal":
                    unseen_err += 1
    def rate(err: int, n: int) -> float:
        return float("nan") if n == 0 else err / n

    return {
        "seen_word_wer": rate(seen_err, seen_n),
        "unseen_word_wer": rate(unseen_err, unseen_n),
        "n_seen_ref_words": seen_n,
        "n_unseen_ref_words": unseen_n,
    }
