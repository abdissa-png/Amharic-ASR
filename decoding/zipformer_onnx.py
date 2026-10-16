"""Streaming Zipformer decode from the ONNX graphs in ``deployment.export_zipformer_streaming``.

One call is the same 77-frame CMVN log-Mel window as ``ZipformerStream``.
The encoder returns the joiner projection, the cache, and CTC logits.
The transducer box is the decoder plus the joiner. The CTC box rescores
``ctc_logits`` with the existing KenLM.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from decoding.zipformer_stream import (
    FEATURE_CHUNK,
    FLUSH_TAIL_S,
    SAMPLE_RATE,
    ZipformerStream,
    _room_noise,
)
from models.features import log_mel
from utils import setup_logging

log = setup_logging()


def _providers() -> list[str]:
    """Use CUDA only when this process can see a GPU.

    ONNX Runtime lists the CUDA provider even when ``cudaSetDevice`` fails,
    and asking for it then prints an error and falls back session by session.
    """
    import onnxruntime as ort

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available and torch.cuda.is_available():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _session(path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), options, providers=_providers())


def _zeros(inp) -> np.ndarray:
    shape = [1 if isinstance(dim, str) or dim is None else int(dim) for dim in inp.shape]
    dtype = np.int64 if "int64" in inp.type else np.float32
    return np.zeros(shape, dtype=dtype)


class OnnxZipformer:
    """Three ONNX Runtime sessions: encoder, decoder, joiner."""

    def __init__(self, encoder: Path, decoder: Path, joiner: Path, *, blank_id: int = 0, unk_id: int = 1) -> None:
        self.encoder_path = Path(encoder)
        self.enc = _session(self.encoder_path)
        self.dec = _session(decoder)
        self.join = _session(joiner)
        self.provider = self.enc.get_providers()[0]
        self.blank_id = int(blank_id)
        self.unk_id = int(unk_id)
        y_shape = self.dec.get_inputs()[0].shape
        self.context_size = int(y_shape[1]) if isinstance(y_shape[1], int) else 2
        self.state_inputs = [inp for inp in self.enc.get_inputs() if inp.name != "x"]
        self.epoch = self._epoch()
        log.info(
            "onnx zipformer %s  provider=%s  states=%d  context=%d",
            self.encoder_path.name,
            self.provider,
            len(self.state_inputs),
            self.context_size,
        )

    def _epoch(self) -> int | None:
        meta_path = self.encoder_path.parent / "meta.json"
        if not meta_path.exists():
            return None
        try:
            return int(json.loads(meta_path.read_text(encoding="utf-8")).get("epoch"))
        except (OSError, ValueError, TypeError):
            return None

    def initial_states(self) -> list[np.ndarray]:
        return [_zeros(inp) for inp in self.state_inputs]

    def initial_context(self) -> np.ndarray:
        context = np.zeros((1, self.context_size), dtype=np.int64)
        if self.context_size > 1:
            context[:, :-1] = -1
        context[:, -1] = self.blank_id
        return context

    def step(
        self,
        feats: np.ndarray,
        states: list[np.ndarray],
        context: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[int], np.ndarray]:
        """``feats`` is ``(1, 77, 80)``. Returns projected frames, CTC logits, states, tokens, context."""
        if feats.shape[1] != FEATURE_CHUNK:
            raise ValueError(f"expected {FEATURE_CHUNK} feature frames, got {feats.shape[1]}")
        feed = {"x": np.ascontiguousarray(feats, dtype=np.float32)}
        for inp, state in zip(self.state_inputs, states):
            feed[inp.name] = state
        out = self.enc.run(None, feed)
        n_states = len(states)
        projected = out[0]
        new_states = [np.ascontiguousarray(value) for value in out[1 : 1 + n_states]]
        ctc = out[1 + n_states]
        tokens: list[int] = []
        for t in range(projected.shape[1]):
            decoder_out = self.dec.run(None, {"y": context})[0]
            logits = self.join.run(
                None,
                {
                    "encoder_out": np.ascontiguousarray(projected[:, t, :]),
                    "decoder_out": decoder_out,
                },
            )[0]
            y = int(np.argmax(logits, axis=-1)[0])
            if y != self.blank_id and y != self.unk_id:
                tokens.append(y)
                nxt = np.array([[y]], dtype=np.int64)
                context = np.concatenate([context[:, 1:], nxt], axis=1)
        return projected, ctc, new_states, tokens, context


class OnnxZipformerStream(ZipformerStream):
    """Mic session with the same audio window as ``ZipformerStream``, stepped in ONNX."""

    def __init__(self, runner: OnnxZipformer, cmvn, device: torch.device) -> None:
        super().__init__(None, cmvn, device)
        self.runner = runner
        self.np_states: list[np.ndarray] | None = None
        self.context_np: np.ndarray | None = None
        self.ctc_logits: torch.Tensor | None = None

    def run_prepared(self, wav: torch.Tensor) -> None:
        """Decode a clip that was already filtered and edge-padded."""
        wav = torch.cat([wav, _room_noise(wav, int(FLUSH_TAIL_S * SAMPLE_RATE))])
        feats = log_mel(wav, dither=0.0)
        if self.cmvn is not None and feats.numel():
            feats = self.cmvn(feats)
        if feats.numel() == 0:
            return
        self._drain(feats, flush=True)
        self.flushed = True

    def _step(self, chunk: torch.Tensor) -> None:
        if self.np_states is None:
            self.np_states = self.runner.initial_states()
            self.context_np = self.runner.initial_context()
        feats = chunk.detach().to(dtype=torch.float32, device="cpu").numpy()
        if feats.ndim == 2:
            feats = feats[None, ...]
        projected, ctc, self.np_states, tokens, self.context_np = self.runner.step(
            feats, self.np_states, self.context_np
        )
        self.tokens.extend(tokens)
        ctc_t = torch.from_numpy(np.ascontiguousarray(ctc))
        proj_t = torch.from_numpy(np.ascontiguousarray(projected))
        self.ctc_logits = ctc_t if self.ctc_logits is None else torch.cat([self.ctc_logits, ctc_t], dim=1)
        self.encoder_out = proj_t if self.encoder_out is None else torch.cat([self.encoder_out, proj_t], dim=1)
