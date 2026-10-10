"""Incremental Zipformer decode with the icefall streaming cache.

A live chunk is ``EMBED_CHUNK`` frames after the convolutional subsample
(32, one of the sizes the causal encoder was trained on). Icefall's
streaming export (``export-onnx-streaming.py``) turns that into feature
frames:

  * the subsample keeps ``(T - 7) // 2`` frames
  * ConvNeXt's kernel is 7, so streaming keeps 3 frames of right context
  * each call therefore reads ``T = EMBED_CHUNK * 2 + 7 + 2 * 3`` frames
  * the next call starts ``EMBED_CHUNK * 2`` frames later (the overlap is
    the right context, not new audio)

The encoder cache is ``Zipformer2.get_init_states`` plus the ConvNeXt left
pad and a count of embed frames already seen. That count masks the zero
left-context at the start of the utterance, same as the export. The
decoder cache is the greedy context (the last ``context_size`` tokens).

Only a full feature window is encoded while the mic is still open. The
tail of the window is real audio, not zeros, so a word that straddles the
chunk edge is not closed early. On stop, room noise is appended and the
remainder is flushed once.
"""

from __future__ import annotations

import time
from typing import List

import numpy as np
import torch
import torchaudio

from decoding.zipformer_greedy import greedy_search_continue, initial_context
from models.features import SAMPLE_RATE, log_mel

# Embed frames per streaming step. Must divide every downsampling factor
# in the streaming configs (1, 2, 4, 8).
EMBED_CHUNK = 32
# 7 from Conv2dSubsampling, 3 ConvNeXt right-context frames times the stride-2.
PAD_FRAMES = 7 + 2 * 3
FEATURE_CHUNK = EMBED_CHUNK * 2 + PAD_FRAMES  # 77
FEATURE_SHIFT = EMBED_CHUNK * 2  # 64
# Output samples dropped so a later resample of a longer clip cannot change
# frames already encoded. Measured for the sinc width used below.
RESAMPLE_GUARD = 256
# Long enough that the last speech frames finish inside a full, unpadded chunk.
FLUSH_TAIL_S = 1.0
_SESSION_TTL_S = 600.0


def init_stream_states(model: torch.nn.Module, device: torch.device) -> List[torch.Tensor]:
    """Encoder caches, ConvNeXt left pad, and embed frames already processed.

    Same order as icefall ``OnnxEncoder.get_init_states``: per-layer
    ``(cached_key, cached_nonlin_attn, cached_val1, cached_val2, cached_conv1,
    cached_conv2)``, then the subsampling pad, then ``processed_lens``.
    """
    states = model.encoder.get_init_states(batch_size=1, device=device)
    states.append(model.encoder_embed.get_init_states(batch_size=1, device=device))
    states.append(torch.zeros(1, dtype=torch.int64, device=device))
    return states


def stream_encoder_chunk(
    model: torch.nn.Module,
    feats: torch.Tensor,
    states: List[torch.Tensor],
) -> tuple[torch.Tensor, List[torch.Tensor]]:
    """Run one cached encoder step.

    ``feats`` is ``(1, FEATURE_CHUNK, 80)``. Returns encoder output
    ``(1, T, C)`` and the updated state list.
    """
    if feats.size(1) != FEATURE_CHUNK:
        raise ValueError(f"expected {FEATURE_CHUNK} feature frames, got {feats.size(1)}")
    left = int(model.encoder.left_context_frames[0])
    if left <= 0:
        raise ValueError("streaming cache needs a fixed left context")
    device = feats.device
    n = feats.size(0)
    x_lens = torch.full((n,), FEATURE_CHUNK, dtype=torch.long, device=device)
    cached_embed = states[-2]
    x, x_lens, new_embed = model.encoder_embed.streaming_forward(
        x=feats,
        x_lens=x_lens,
        cached_left_pad=cached_embed,
    )
    if int(x.size(1)) != EMBED_CHUNK:
        raise RuntimeError(f"embed chunk produced {x.size(1)} frames, expected {EMBED_CHUNK}")

    processed = states[-1]
    # Mask the zero-filled left context until real frames have replaced it.
    # Newest cached frames sit on the right, next to this chunk.
    processed_mask = torch.arange(left, device=device).expand(n, left)
    processed_mask = (processed.unsqueeze(1) <= processed_mask).to(torch.int32).flip(1)
    current = torch.zeros(n, EMBED_CHUNK, dtype=torch.int32, device=device)
    pad_mask = torch.cat([processed_mask, current], dim=1)
    new_processed = processed + x_lens

    x = x.permute(1, 0, 2)
    encoder_out, _out_lens, new_encoder_states = model.encoder.streaming_forward(
        x=x,
        x_lens=x_lens,
        states=states[:-2],
        src_key_padding_mask=pad_mask,
    )
    encoder_out = encoder_out.permute(1, 0, 2)
    return encoder_out, new_encoder_states + [new_embed, new_processed]


def _resample_16k(samples: torch.Tensor, sample_rate: int) -> torch.Tensor:
    if int(sample_rate) == SAMPLE_RATE:
        return samples
    return torchaudio.functional.resample(
        samples.unsqueeze(0),
        int(sample_rate),
        SAMPLE_RATE,
        lowpass_filter_width=64,
        rolloff=0.9475,
        resampling_method="sinc_interp_kaiser",
    ).squeeze(0)


def stable_16k(samples: torch.Tensor, sample_rate: int, *, final: bool) -> torch.Tensor:
    """16 kHz audio whose prefix will not change when more input arrives.

    On stop, ``final`` keeps the filter tail. Until then the last
    ``RESAMPLE_GUARD`` output samples are held back.
    """
    wav = _resample_16k(samples, sample_rate)
    if final or int(sample_rate) == SAMPLE_RATE:
        return wav
    if wav.numel() <= RESAMPLE_GUARD:
        return wav.new_zeros(0)
    return wav[:-RESAMPLE_GUARD].contiguous()


def _bandlimit(wav: torch.Tensor) -> torch.Tensor:
    """80 Hz–7.6 kHz, matching ``demo.app._prep_waveform`` without a global mean."""
    if wav.numel() == 0:
        return wav
    wav = torchaudio.functional.highpass_biquad(
        wav.unsqueeze(0), SAMPLE_RATE, cutoff_freq=80.0
    ).squeeze(0)
    return torchaudio.functional.lowpass_biquad(
        wav.unsqueeze(0), SAMPLE_RATE, cutoff_freq=7600.0
    ).squeeze(0)


def _speech_rms(wav: torch.Tensor) -> float:
    if wav.numel() == 0:
        return 0.0
    hop = max(1, SAMPLE_RATE // 50)
    if wav.numel() < hop * 4:
        return float(wav.pow(2).mean().sqrt())
    frames = wav.unfold(0, hop, hop)
    energy = frames.pow(2).mean(-1)
    keep = energy >= energy.median()
    if not bool(keep.any()):
        return float(wav.pow(2).mean().sqrt())
    return float(frames[keep].pow(2).mean().sqrt())


def _stream_gain(wav: torch.Tensor) -> float:
    """Same quiet-mic boost as ``_prep_waveform``, frozen for the session."""
    rms = _speech_rms(wav)
    if rms < 0.02:
        return min(8.0, 0.06 / max(rms, 1e-8))
    return 1.0


def _room_noise(wav: torch.Tensor, n: int) -> torch.Tensor:
    """``n`` samples from the quietest 10 ms frames. Same construction as the demo tail."""
    hop = max(1, SAMPLE_RATE // 100)
    frames = wav[: wav.numel() // hop * hop].reshape(-1, hop)
    if n <= 0 or frames.size(0) < 4:
        return wav.new_zeros(max(0, n))
    quiet = frames[frames.pow(2).mean(-1).argsort()[: max(4, frames.size(0) // 10)]].reshape(-1)
    return quiet.repeat(n // quiet.numel() + 1)[:n]


class ZipformerStream:
    """One microphone session. Audio is append-only; chunks are not recomputed."""

    def __init__(self, model: torch.nn.Module, cmvn, device: torch.device) -> None:
        self.model = model
        self.cmvn = cmvn
        self.device = device
        self.native = torch.empty(0, dtype=torch.float32)
        self.n = 0
        self.sr: int | None = None
        self.gain: float | None = None
        self.states: List[torch.Tensor] | None = None
        self.context: torch.Tensor | None = None
        self.tokens: list[int] = []
        self.encoder_out: torch.Tensor | None = None
        self.feat_done = 0
        self.feats: torch.Tensor | None = None
        self.flushed = False
        self.touched = time.monotonic()

    def add(
        self,
        samples: np.ndarray | torch.Tensor,
        sample_rate: int,
        offset: int,
        *,
        final: bool = False,
    ) -> int:
        """Append samples that start at ``offset`` in the recording. Returns the count kept.

        A retry that overlaps audio already accepted is ignored up to ``self.n``.
        ``offset`` past that is a gap.
        """
        self.touched = time.monotonic()
        if self.flushed:
            return self.n
        offset = int(offset)
        if offset < 0:
            raise ValueError(f"bad audio offset {offset}")
        sample_rate = int(sample_rate)
        if self.sr is None:
            self.sr = sample_rate
        elif sample_rate != self.sr:
            raise ValueError("sample rate changed mid-recording")
        if offset > self.n:
            raise ValueError(f"audio gap at sample {offset}; session has {self.n}")

        incoming = _as_float(samples)
        skip = self.n - offset
        if skip > 0:
            incoming = incoming[skip:] if skip < incoming.numel() else incoming[:0]
        if incoming.numel():
            self.native = incoming if self.native.numel() == 0 else torch.cat([self.native, incoming])
            self.n += int(incoming.numel())
        elif not final:
            return self.n

        if self.native.numel() == 0:
            if final:
                self.flushed = True
            return self.n

        wav = stable_16k(self.native, self.sr, final=final)
        prepared = self._prepare(wav, final=final)
        if self.gain is None:
            return self.n
        if final:
            prepared = torch.cat([prepared, _room_noise(prepared, int(FLUSH_TAIL_S * SAMPLE_RATE))])
        feats = self._featurize(prepared)
        if feats.numel() == 0:
            if final:
                self.flushed = True
            return self.n
        self._drain(feats, flush=final)
        if final:
            self.flushed = True
        return self.n

    def _featurize(self, wav: torch.Tensor) -> torch.Tensor:
        """Log-mel rows. Frames already read by the encoder stay put.

        Torchaudio's fbank shifts finished frames by about 1e-6 when the clip
        grows. Recomputing them would disagree with the encoder cache, so the
        tail is the only part that updates.
        """
        fresh = log_mel(wav, dither=0.0)
        if fresh.numel() == 0:
            return self.feats if self.feats is not None else fresh
        if self.cmvn is not None:
            fresh = self.cmvn(fresh)
        if self.feats is None or self.encoder_out is None:
            self.feats = fresh
            return self.feats
        # Previous window is [feat_done - shift, feat_done + chunk - shift).
        keep = min(
            self.feat_done + FEATURE_CHUNK - FEATURE_SHIFT,
            self.feats.size(0),
            fresh.size(0),
        )
        self.feats = torch.cat([self.feats[:keep], fresh[keep:]], dim=0)
        return self.feats

    def _prepare(self, wav: torch.Tensor, *, final: bool) -> torch.Tensor:
        filtered = _bandlimit(wav)
        if self.gain is None and (final or filtered.numel() >= SAMPLE_RATE // 2):
            self.gain = _stream_gain(filtered)
        if self.gain is None:
            return filtered
        return (filtered * self.gain).clamp(-1.0, 1.0)

    def _drain(self, feats: torch.Tensor, *, flush: bool) -> None:
        total = int(feats.size(0))
        while self.feat_done + FEATURE_CHUNK <= total:
            self._step(feats[self.feat_done : self.feat_done + FEATURE_CHUNK])
            self.feat_done += FEATURE_SHIFT
        if flush and self.feat_done < total:
            tail = feats[self.feat_done :]
            pad = FEATURE_CHUNK - int(tail.size(0))
            if pad > 0:
                tail = torch.cat([tail, tail.new_zeros(pad, tail.size(1))], dim=0)
            self._step(tail)
            self.feat_done = total

    def _step(self, chunk: torch.Tensor) -> None:
        if self.states is None:
            self.states = init_stream_states(self.model, self.device)
            self.context = initial_context(self.model)
        feats = chunk.unsqueeze(0).to(self.device)
        encoder_out, self.states = stream_encoder_chunk(self.model, feats, self.states)
        tokens, self.context = greedy_search_continue(self.model, encoder_out, self.context)
        self.tokens.extend(tokens)
        self.encoder_out = (
            encoder_out if self.encoder_out is None else torch.cat([self.encoder_out, encoder_out], dim=1)
        )


def _as_float(samples: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(samples, torch.Tensor):
        return samples.detach().to(dtype=torch.float32, device="cpu").reshape(-1).contiguous()
    array = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
    return torch.from_numpy(array)


def prune_streams(sessions: dict[str, ZipformerStream], now: float | None = None) -> None:
    now = time.monotonic() if now is None else now
    stale = [key for key, sess in sessions.items() if now - sess.touched > _SESSION_TTL_S]
    for key in stale:
        del sessions[key]
