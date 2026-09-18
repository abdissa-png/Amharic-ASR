"""WER / CER / MER and Dataset.ET-comparable scoring."""

from __future__ import annotations

from typing import Iterable, Sequence

import jiwer

from evaluation.geez_eval import normalize as geez_normalize


def _pairs(refs: Sequence[str], hyps: Sequence[str], *, fold: bool) -> tuple[list[str], list[str]]:
    r = [geez_normalize(x, fold_geez=fold) for x in refs]
    h = [geez_normalize(x, fold_geez=fold) for x in hyps]
    pairs = [(a, b) for a, b in zip(r, h, strict=True) if a.strip()]
    if not pairs:
        return [], []
    return [a for a, _ in pairs], [b for _, b in pairs]


def mer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    """Match Error Rate: (S+D+I) / (S+D+I+C)."""
    out = jiwer.process_words(list(refs), list(hyps))
    denom = out.substitutions + out.deletions + out.insertions + out.hits
    if denom == 0:
        return 0.0
    return (out.substitutions + out.deletions + out.insertions) / denom


def score_texts(
    refs: Sequence[str],
    hyps: Sequence[str],
    *,
    fold_geez: bool = True,
) -> dict[str, float]:
    r, h = _pairs(refs, hyps, fold=fold_geez)
    if not r:
        return {
            "wer": float("nan"),
            "cer": float("nan"),
            "mer": float("nan"),
            "n": 0,
            "substitutions": 0,
            "deletions": 0,
            "insertions": 0,
            "hits": 0,
        }
    counted = jiwer.process_words(r, h)
    ref_n = counted.substitutions + counted.deletions + counted.hits
    err = counted.substitutions + counted.deletions + counted.insertions
    wer = err / ref_n if ref_n else 0.0
    mer_v = err / (ref_n + counted.insertions) if (ref_n + counted.insertions) else 0.0
    cer = float(jiwer.cer(r, h))
    return {
        "wer": wer,
        "cer": cer,
        "mer": mer_v,
        "wer_pct": round(wer * 100, 2),
        "cer_pct": round(cer * 100, 2),
        "mer_pct": round(mer_v * 100, 2),
        "n": len(r),
        "substitutions": int(counted.substitutions),
        "deletions": int(counted.deletions),
        "insertions": int(counted.insertions),
        "hits": int(counted.hits),
        "empty_hyp_rate": sum(1 for x in h if not x.strip()) / len(h),
    }


def alignment_ops(ref: str, hyp: str) -> list[tuple[str, str | None, str | None]]:
    """Word-level alignment ops: equal / substitute / delete / insert."""
    out = jiwer.process_words([ref], [hyp])
    ops: list[tuple[str, str | None, str | None]] = []
    align = out.alignments[0]
    ref_w = ref.split()
    hyp_w = hyp.split()
    for chunk in align:
        kind = str(getattr(chunk, "type", chunk)).lower().split(".")[-1]
        rs, re = chunk.ref_start_idx, chunk.ref_end_idx
        hs, he = chunk.hyp_start_idx, chunk.hyp_end_idx
        if kind == "equal":
            for i, j in zip(range(rs, re), range(hs, he)):
                ops.append(("equal", ref_w[i], hyp_w[j]))
        elif kind == "substitute":
            n = max(re - rs, he - hs)
            for k in range(n):
                rw = ref_w[rs + k] if rs + k < re else None
                hw = hyp_w[hs + k] if hs + k < he else None
                if rw is None:
                    ops.append(("insert", None, hw))
                elif hw is None:
                    ops.append(("delete", rw, None))
                else:
                    ops.append(("substitute", rw, hw))
        elif kind == "delete":
            for i in range(rs, re):
                ops.append(("delete", ref_w[i], None))
        elif kind in ("insert", "insertion"):
            for j in range(hs, he):
                ops.append(("insert", None, hyp_w[j]))
    return ops
