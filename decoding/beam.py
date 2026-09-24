"""CTC prefix beam search (Decoder B).

The previous implementation called ``torch.topk`` and copied prefix tuples
on every frame. That is hours of Python on Dataset.ET. This version keeps
prefixes in a trie (O(1) extend, no tuple copies) and vectorizes top-k.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

NEG = float("-inf")


def _logaddexp(a: float, b: float) -> float:
    if a == NEG:
        return b
    if b == NEG:
        return a
    m = a if a > b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))


class _Node:
    __slots__ = ("token", "parent", "pb", "pnb", "children")

    def __init__(self, token: Optional[int], parent: Optional["_Node"]) -> None:
        self.token = token
        self.parent = parent
        self.pb = NEG
        self.pnb = NEG
        self.children: dict[int, _Node] = {}


def _child(node: _Node, tok: int) -> _Node:
    ch = node.children.get(tok)
    if ch is None:
        ch = _Node(tok, node)
        node.children[tok] = ch
    return ch


def prefix_beam_search(
    log_probs: torch.Tensor,
    tokenizer,
    *,
    beam_size: int = 16,
    blank_id: int = 0,
    token_top_k: int = 8,
) -> str:
    """Single-utterance prefix beam search. ``log_probs``: (T, V)."""
    lp = log_probs.detach()
    if lp.device.type != "cpu":
        lp = lp.cpu()
    lp = lp.float().contiguous()
    t_len, vocab = lp.shape
    k = min(token_top_k, vocab)
    topv, topi = torch.topk(lp, k=k, dim=-1)
    topv_np = topv.numpy()
    topi_np = topi.numpy()
    blank_np = lp[:, blank_id].numpy()

    root = _Node(None, None)
    root.pb = 0.0
    beam = [root]

    for t in range(t_len):
        ids = topi_np[t]
        vals = topv_np[t]
        blank_logp = float(blank_np[t])
        pending: dict[_Node, list[float]] = {}

        def add(node: _Node, is_blank: bool, score: float) -> None:
            rec = pending.get(node)
            if rec is None:
                rec = [NEG, NEG]
                pending[node] = rec
            idx = 0 if is_blank else 1
            rec[idx] = _logaddexp(rec[idx], score)

        for node in beam:
            p_tot = _logaddexp(node.pb, node.pnb)
            last = node.token
            add(node, True, p_tot + blank_logp)
            for j in range(k):
                c = int(ids[j])
                if c == blank_id:
                    continue
                logp = float(vals[j])
                if last == c:
                    add(_child(node, c), False, node.pb + logp)
                    add(node, False, node.pnb + logp)
                else:
                    add(_child(node, c), False, p_tot + logp)

        scored = [(_logaddexp(pb, pnb), node, pb, pnb) for node, (pb, pnb) in pending.items()]
        scored.sort(key=lambda x: x[0], reverse=True)
        beam = []
        for _, node, pb, pnb in scored[:beam_size]:
            node.pb = pb
            node.pnb = pnb
            beam.append(node)

    best = max(beam, key=lambda n: _logaddexp(n.pb, n.pnb))
    toks: list[int] = []
    node = best
    while node is not None and node.token is not None:
        toks.append(node.token)
        node = node.parent
    toks.reverse()
    return tokenizer.decode(toks, collapse=False)


def beam_decode_batch(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    tokenizer,
    *,
    beam_size: int = 16,
    token_top_k: int = 8,
) -> list[str]:
    log_probs = F.log_softmax(logits, dim=-1).detach().float().cpu()
    hyps = []
    for b in range(log_probs.size(0)):
        t = int(lengths[b])
        hyps.append(
            prefix_beam_search(
                log_probs[b, :t],
                tokenizer,
                beam_size=beam_size,
                blank_id=tokenizer.blank_id,
                token_top_k=token_top_k,
            )
        )
    return hyps
