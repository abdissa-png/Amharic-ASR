"""Greedy CTC decode (collapse repeats, drop blank)."""

from __future__ import annotations

from typing import Sequence

import torch


def greedy_ids(logits: torch.Tensor) -> list[list[int]]:
    """logits: (B, T, V) → per-utterance best-path ids (no collapse)."""
    return logits.argmax(-1).cpu().tolist()


def collapse_ctc(ids: Sequence[int], blank_id: int = 0) -> list[int]:
    out: list[int] = []
    prev = None
    for i in ids:
        i = int(i)
        if i == blank_id:
            prev = blank_id
            continue
        if i == prev:
            continue
        out.append(i)
        prev = i
    return out


def greedy_decode(logits: torch.Tensor, tokenizer, lengths: torch.Tensor | None = None) -> list[str]:
    hyps = []
    ids = logits.argmax(-1)
    for b in range(ids.size(0)):
        seq = ids[b]
        if lengths is not None:
            seq = seq[: int(lengths[b])]
        collapsed = collapse_ctc(seq.tolist(), tokenizer.blank_id)
        hyps.append(tokenizer.decode(collapsed, collapse=False))
    return hyps
