"""CTC prefix beam with a character or morpheme KenLM.

pyctcdecode scores *words*. A character n-gram has to update KenLM on every
emitted letter, with ``▁`` as the space token. A morpheme n-gram is scored
when a space closes a word, after ``surface_morphs`` peels prefixes/suffixes.

    python -m decoding.train_lm --unit char --lm-only --order 6 \\
        --src-text exp/20m_char_12layer/lm_word/train.txt \\
        --out-dir exp/zipformer_s_streaming_pool/lm_char \\
        --prune "0 0 0 1 1 2"
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from decoding.kenlm import _ensure_kenlm_bindings, lm_files
from decoding.train_lm import SPACE_SYM
from phonology.decomposition import surface_morphs

NEG = float("-inf")
LN10 = math.log(10.0)
EOS = "</s>"


def _logaddexp(a: float, b: float) -> float:
    if a == NEG:
        return b
    if b == NEG:
        return a
    m = a if a > b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))


class _Node:
    __slots__ = (
        "token",
        "parent",
        "pb",
        "pnb",
        "children",
        "lm_state",
        "lm_log10",
        "n_words",
        "cur_word",
    )

    def __init__(self, token: Optional[int], parent: Optional["_Node"]) -> None:
        self.token = token
        self.parent = parent
        self.pb = NEG
        self.pnb = NEG
        self.children: dict[int, _Node] = {}
        self.lm_state = None
        self.lm_log10 = 0.0
        self.n_words = 0
        self.cur_word = ""


def _child(node: _Node, tok: int) -> _Node:
    ch = node.children.get(tok)
    if ch is None:
        ch = _Node(tok, node)
        node.children[tok] = ch
    return ch


def load_unit_model(lm_dir: str | Path):
    path = Path(lm_dir)
    model_path, _ = lm_files(path) if path.is_dir() else (path, None)
    if model_path is None or not model_path.exists():
        raise FileNotFoundError(f"no KenLM model at {lm_dir}")
    km = _ensure_kenlm_bindings()
    return km, km.Model(str(model_path)), model_path


def _char_table(tokenizer) -> list[str | None]:
    """KenLM token per CTC id. None means skip (blank/pad)."""
    table: list[str | None] = []
    for i, s in enumerate(tokenizer.tokens):
        if i == tokenizer.blank_id or s in ("<blank>", "<pad>"):
            table.append(None)
        elif s == "<unk>":
            table.append("<unk>")
        elif s == " ":
            table.append(SPACE_SYM)
        else:
            table.append(s)
    return table


def _score_tokens(km, model, state, log10: float, tokens: list[str]):
    for tok in tokens:
        nxt = km.State()
        log10 += float(model.BaseScore(state, tok, nxt))
        state = nxt
    return state, log10


def _advance_char(km, model, parent: _Node, tok_id: int, table: list[str | None]) -> _Node:
    child = _child(parent, tok_id)
    if child.lm_state is not None:
        return child
    piece = table[tok_id] if 0 <= tok_id < len(table) else "<unk>"
    if piece is None:
        piece = "<unk>"
    child.lm_state, child.lm_log10 = _score_tokens(km, model, parent.lm_state, parent.lm_log10, [piece])
    child.n_words = parent.n_words + int(piece == SPACE_SYM)
    if piece == SPACE_SYM:
        child.cur_word = ""
    else:
        child.cur_word = parent.cur_word + ("" if piece == "<unk>" else piece)
    return child


def tokenizer_char(table: list[str | None], tok_id: int) -> str:
    piece = table[tok_id] if 0 <= tok_id < len(table) else ""
    if piece is None or piece == "<unk>":
        return ""
    if piece == SPACE_SYM:
        return " "
    return piece


def _advance_morph(km, model, parent: _Node, tok_id: int, table: list[str | None]) -> _Node:
    child = _child(parent, tok_id)
    if child.lm_state is not None:
        return child
    piece = table[tok_id] if 0 <= tok_id < len(table) else "<unk>"
    is_space = piece == SPACE_SYM
    if is_space:
        morphs = surface_morphs(parent.cur_word) if parent.cur_word else []
        child.lm_state, child.lm_log10 = _score_tokens(
            km, model, parent.lm_state, parent.lm_log10, morphs + [SPACE_SYM]
        )
        child.n_words = parent.n_words + 1
        child.cur_word = ""
    else:
        child.lm_state = parent.lm_state
        child.lm_log10 = parent.lm_log10
        child.n_words = parent.n_words
        ch = tokenizer_char(table, tok_id)
        child.cur_word = parent.cur_word + ch
    return child


def _final_lm(km, model, node: _Node, unit: str) -> float:
    state, log10 = node.lm_state, node.lm_log10
    if unit == "morph" and node.cur_word:
        morphs = surface_morphs(node.cur_word)
        state, log10 = _score_tokens(km, model, state, log10, morphs)
    state, log10 = _score_tokens(km, model, state, log10, [EOS])
    return log10


def prefix_beam_unit_lm(
    log_probs: torch.Tensor,
    tokenizer,
    km,
    model,
    *,
    unit: str = "char",
    beam_size: int = 32,
    blank_id: int = 0,
    token_top_k: int = 12,
    alpha: float = 0.6,
    beta: float = 0.8,
) -> str:
    """Single-utterance CTC prefix beam with incremental KenLM. ``log_probs``: (T, V)."""
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
    table = _char_table(tokenizer)
    advance = _advance_char if unit == "char" else _advance_morph

    root = _Node(None, None)
    root.pb = 0.0
    root.lm_state = km.State()
    model.BeginSentenceWrite(root.lm_state)
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
                if 0 <= c < len(table) and table[c] is None:
                    continue
                logp = float(vals[j])
                if last == c:
                    add(advance(km, model, node, c, table), False, node.pb + logp)
                    add(node, False, node.pnb + logp)
                else:
                    add(advance(km, model, node, c, table), False, p_tot + logp)

        scored = []
        for node, (pb, pnb) in pending.items():
            acoustic = _logaddexp(pb, pnb)
            total = acoustic + alpha * LN10 * node.lm_log10 + beta * node.n_words
            scored.append((total, node, pb, pnb))
        scored.sort(key=lambda x: x[0], reverse=True)
        beam = []
        for _, node, pb, pnb in scored[:beam_size]:
            node.pb = pb
            node.pnb = pnb
            beam.append(node)

    best = max(
        beam,
        key=lambda n: _logaddexp(n.pb, n.pnb) + alpha * LN10 * _final_lm(km, model, n, unit) + beta * n.n_words,
    )
    toks: list[int] = []
    node = best
    while node is not None and node.token is not None:
        toks.append(node.token)
        node = node.parent
    toks.reverse()
    return tokenizer.decode(toks, collapse=False)


def decode_batch(
    logits,
    lengths,
    tokenizer,
    km,
    model,
    *,
    unit: str = "char",
    beam_size: int = 32,
    token_top_k: int = 12,
    alpha: float = 0.6,
    beta: float = 0.8,
) -> list[str]:
    log_probs = F.log_softmax(logits, dim=-1).detach().float().cpu()
    hyps = []
    for b in range(log_probs.size(0)):
        t = int(lengths[b])
        hyps.append(
            prefix_beam_unit_lm(
                log_probs[b, :t],
                tokenizer,
                km,
                model,
                unit=unit,
                beam_size=beam_size,
                blank_id=tokenizer.blank_id,
                token_top_k=token_top_k,
                alpha=alpha,
                beta=beta,
            )
        )
    return hyps
