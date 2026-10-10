"""Greedy transducer decode, one symbol per frame.

Copied from icefall
``egs/librispeech/ASR/pruned_transducer_stateless2/beam_search.py``
lines 659-781 at commit 3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1.
``egs/librispeech/ASR/zipformer/beam_search.py`` is a symlink to that file.
``zipformer/decode.py`` calls ``greedy_search_batch`` when
``--decoding-method greedy_search`` and ``--max-sym-per-frame 1``.

The blank penalty is applied to index 0, which is the blank id in that
recipe and in this repo's character tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from torch import nn


@dataclass
class DecodingResults:
    timestamps: List[List[int]]
    hyps: List[List[int]]
    scores: Optional[List[List[float]]] = None


def greedy_search_batch(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    blank_penalty: float = 0,
    return_timestamps: bool = False,
) -> Union[List[List[int]], DecodingResults]:
    """Greedy search in batch mode. It hardcodes --max-sym-per-frame=1.
    Args:
      model:
        The transducer model.
      encoder_out:
        Output from the encoder. Its shape is (N, T, C), where N >= 1.
      encoder_out_lens:
        A 1-D tensor of shape (N,), containing number of valid frames in
        encoder_out before padding.
      return_timestamps:
        Whether to return timestamps.
    Returns:
      If return_timestamps is False, return the decoded result.
      Else, return a DecodingResults object containing
      decoded result and corresponding timestamps.
    """
    assert encoder_out.ndim == 3
    assert encoder_out.size(0) >= 1, encoder_out.size(0)

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    device = next(model.parameters()).device

    blank_id = model.decoder.blank_id
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.decoder.context_size

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)
    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    # Context is the icefall prefix: context_size-1 negative pads, then blank.
    # Rows stay in packed (longest-first) order until the single host copy below.
    context = torch.zeros(N, context_size, device=device, dtype=torch.int64)
    if context_size > 1:
        context[:, :-1] = -1
    context[:, -1] = blank_id

    decoder_out = model.joiner.decoder_proj(model.decoder(context, need_pad=False))
    # decoder_out: (N, 1, decoder_out_dim)

    encoder_proj = model.joiner.encoder_proj(packed_encoder_out.data)

    # -1 is blank, unk, or a finished row. Token ids other than blank are >= 1.
    max_t = len(batch_size_list)
    token_at = torch.full((max_t, N), -1, device=device, dtype=torch.long)
    score_at = torch.zeros((max_t, N), device=device, dtype=torch.float32) if return_timestamps else None

    offset = 0
    for t, batch_size in enumerate(batch_size_list):
        start = offset
        end = offset + batch_size
        current_encoder_out = encoder_proj.data[start:end].unsqueeze(1).unsqueeze(1)
        # current_encoder_out's shape: (batch_size, 1, 1, encoder_out_dim)
        offset = end

        logits = model.joiner(
            current_encoder_out,
            decoder_out[:batch_size].unsqueeze(1),
            project_input=False,
        )
        # logits'shape (batch_size, 1, 1, vocab_size)
        logits = logits.squeeze(1).squeeze(1)  # (batch_size, vocab_size)
        assert logits.ndim == 2, logits.shape

        if blank_penalty != 0:
            logits = logits.clone()
            logits[:, 0] = logits[:, 0] - blank_penalty

        y = logits.argmax(dim=1)
        emit = (y != blank_id) & (y != unk_id)
        token_at[t, :batch_size] = torch.where(emit, y, token_at[t, :batch_size])
        if score_at is not None:
            chosen = logits.gather(1, y.unsqueeze(1)).squeeze(1)
            score_at[t, :batch_size] = torch.where(emit, chosen, score_at[t, :batch_size])

        # Same update as rebuilding the context from the Python hyp, without
        # copying the argmax back to the host. Blank and unk leave the row as it was.
        active = context[:batch_size]
        shifted = torch.cat([active[:, 1:], y.unsqueeze(1)], dim=1)
        context[:batch_size] = torch.where(emit.unsqueeze(1), shifted, active)
        decoder_out = model.joiner.decoder_proj(
            model.decoder(context[:batch_size], need_pad=False)
        )

    token_cpu = token_at.detach().cpu()
    score_cpu = score_at.detach().cpu() if score_at is not None else None
    sorted_lens = (
        encoder_out_lens.detach().cpu()[packed_encoder_out.sorted_indices.cpu()].tolist()
    )
    sorted_ans = []
    sorted_timestamps = []
    sorted_scores = []
    for i, length in enumerate(sorted_lens):
        row = token_cpu[: int(length), i].tolist()
        sorted_ans.append([int(v) for v in row if int(v) >= 0])
        if score_cpu is not None:
            sorted_timestamps.append([t for t, v in enumerate(row) if int(v) >= 0])
            sorted_scores.append(
                [float(score_cpu[t, i]) for t, v in enumerate(row) if int(v) >= 0]
            )

    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    ans = [sorted_ans[unsorted_indices[i]] for i in range(N)]
    ans_timestamps = []
    ans_scores = []
    if score_cpu is not None:
        ans_timestamps = [sorted_timestamps[unsorted_indices[i]] for i in range(N)]
        ans_scores = [sorted_scores[unsorted_indices[i]] for i in range(N)]

    if not return_timestamps:
        return ans
    else:
        return DecodingResults(
            hyps=ans,
            timestamps=ans_timestamps,
            scores=ans_scores,
        )


def initial_context(model: nn.Module, batch_size: int = 1) -> torch.Tensor:
    """Blank context used at the start of ``greedy_search_batch``."""
    device = next(model.parameters()).device
    context_size = model.decoder.context_size
    blank_id = model.decoder.blank_id
    context = torch.zeros(batch_size, context_size, device=device, dtype=torch.int64)
    if context_size > 1:
        context[:, :-1] = -1
    context[:, -1] = blank_id
    return context


def greedy_search_continue(
    model: nn.Module,
    encoder_out: torch.Tensor,
    context: torch.Tensor,
    blank_penalty: float = 0,
) -> tuple[list[int], torch.Tensor]:
    """Greedy-decode new encoder frames, continuing ``context``.

    ``encoder_out`` is ``(1, T, C)`` for frames the encoder has not decoded yet.
    ``context`` is ``(1, context_size)``, the same layout ``greedy_search_batch``
    keeps. Blank and unk leave the context in place. Returns the token ids
    emitted by these frames, and the context to pass with the next chunk.
    """
    assert encoder_out.ndim == 3 and encoder_out.size(0) == 1, encoder_out.shape
    assert context.ndim == 2 and context.size(0) == 1, context.shape
    if encoder_out.size(1) == 0:
        return [], context

    blank_id = model.decoder.blank_id
    unk_id = getattr(model, "unk_id", blank_id)
    encoder_proj = model.joiner.encoder_proj(encoder_out)
    decoder_out = model.joiner.decoder_proj(model.decoder(context, need_pad=False))
    tokens: list[int] = []
    for t in range(encoder_out.size(1)):
        current = encoder_proj[:, t : t + 1, :].unsqueeze(2)
        logits = model.joiner(current, decoder_out.unsqueeze(1), project_input=False)
        logits = logits.squeeze(1).squeeze(1)
        if blank_penalty != 0:
            logits = logits.clone()
            logits[:, 0] = logits[:, 0] - blank_penalty
        y = logits.argmax(dim=1)
        emit = bool((y != blank_id).item() and (y != unk_id).item())
        if not emit:
            continue
        tokens.append(int(y.item()))
        context = torch.cat([context[:, 1:], y.view(1, 1)], dim=1)
        decoder_out = model.joiner.decoder_proj(model.decoder(context, need_pad=False))
    return tokens, context
