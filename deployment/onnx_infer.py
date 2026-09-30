"""ONNX Runtime inference for a Conformer CTC export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def available_providers() -> list[str]:
    import onnxruntime as ort

    return list(ort.get_available_providers())


def select_providers(prefer: str = "auto") -> list:
    """``auto``: CUDA then CPU. ``cuda`` requires CUDA EP. ``cpu`` forces CPU."""
    import onnxruntime as ort

    have = set(ort.get_available_providers())
    prefer = (prefer or "auto").lower()
    cpu = "CPUExecutionProvider"
    cuda = "CUDAExecutionProvider"
    cuda_opts = {
        "device_id": 0,
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_algo_search": "HEURISTIC",
        "do_copy_in_default_stream": True,
    }
    if prefer == "cpu":
        return [cpu]
    if prefer == "cuda":
        if cuda not in have:
            raise RuntimeError(
                "CUDAExecutionProvider is not in this ONNX Runtime build. "
                "Install onnxruntime-gpu (and uninstall the CPU-only onnxruntime wheel)."
            )
        return [(cuda, cuda_opts), cpu]
    if cuda in have:
        return [(cuda, cuda_opts), cpu]
    return [cpu] if cpu in have else list(have)


class OnnxCTC:
    """Drop-in ``(feats, lengths) -> (logits, logit_lengths)`` for evaluate.py."""

    def __init__(
        self,
        path: str | Path,
        providers: list[str] | None = None,
        prefer: str = "auto",
    ) -> None:
        import onnxruntime as ort

        self.path = Path(path)
        if providers is None:
            providers = select_providers(prefer)
        self.requested = list(providers)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        try:
            self.session = ort.InferenceSession(str(self.path), so, providers=providers)
        except Exception:
            names = [p[0] if isinstance(p, tuple) else p for p in providers]
            if names == ["CPUExecutionProvider"]:
                raise
            self.session = ort.InferenceSession(str(self.path), so, providers=["CPUExecutionProvider"])
        self.providers = self.session.get_providers()
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, feats: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feeds = {
            "feats": np.ascontiguousarray(feats.detach().cpu().numpy(), dtype=np.float32),
            "lengths": np.ascontiguousarray(lengths.detach().cpu().numpy(), dtype=np.int64),
        }
        logits, logit_lens = self.session.run(self.output_names, feeds)
        return torch.from_numpy(np.array(logits)), torch.from_numpy(np.array(logit_lens))
