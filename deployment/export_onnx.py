"""ONNX export of a trained Conformer CTC checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from models.ctc import build_model
from train import load_tokenizer
from utils import ROOT, load_yaml, setup_logging

log = setup_logging()


class ExportWrapper(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor):
        logits, logit_lens = self.model(feats, lengths)
        return logits, logit_lens


class ExplicitMHA(nn.Module):
    """Math-path self-attention with the same weights as ``nn.MultiheadAttention``.

    The fused MHA kernel traces into ONNX incorrectly (cosine ~0.90 vs PyTorch).
    """

    def __init__(self, mha: nn.MultiheadAttention) -> None:
        super().__init__()
        self.num_heads = mha.num_heads
        self.embed_dim = mha.embed_dim
        self.head_dim = mha.head_dim
        self.in_proj_weight = mha.in_proj_weight
        self.in_proj_bias = mha.in_proj_bias
        self.out_proj = mha.out_proj
        self.scale = float(self.head_dim) ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ):
        del key, value, need_weights
        b, t, _ = query.shape
        qkv = torch.nn.functional.linear(query, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q * self.scale, k.transpose(-2, -1))
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, self.embed_dim)
        return self.out_proj(out), None


def rewrite_attention_for_export(model: nn.Module) -> int:
    to_fix = [mod for mod in model.modules() if isinstance(getattr(mod, "attn", None), nn.MultiheadAttention)]
    for mod in to_fix:
        mod.attn = ExplicitMHA(mod.attn)
    return len(to_fix)


def _disable_mha_fastpath() -> None:
    try:
        torch.backends.mha.set_fastpath_enabled(False)
    except Exception:
        pass


def _verify(wrapped: nn.Module, out: Path, feats: torch.Tensor, lengths: torch.Tensor) -> None:
    import numpy as np
    import onnxruntime as ort

    with torch.no_grad():
        pt_logits, pt_lens = wrapped(feats, lengths)
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    onnx_logits, onnx_lens = sess.run(
        None,
        {
            "feats": np.ascontiguousarray(feats.detach().cpu().numpy(), dtype=np.float32),
            "lengths": np.ascontiguousarray(lengths.detach().cpu().numpy(), dtype=np.int64),
        },
    )
    a = pt_logits.detach().cpu().numpy().reshape(-1)
    b = np.asarray(onnx_logits).reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    cosine = float(np.dot(a, b) / denom)
    max_abs = float(np.max(np.abs(a - b)))
    lens_ok = bool(np.array_equal(pt_lens.cpu().numpy(), np.asarray(onnx_lens)))
    log.info(
        "onnx vs pytorch  cosine=%.6f  max_abs=%.4g  lengths_match=%s  shape=%s",
        cosine,
        max_abs,
        lens_ok,
        tuple(pt_logits.shape),
    )
    if cosine < 0.999 or not lens_ok:
        raise RuntimeError(f"ONNX mismatch: cosine={cosine:.6f} lengths_match={lens_ok}")


def export(config: Path, ckpt: Path, out: Path) -> Path:
    cfg = load_yaml(config)
    exp_dir = ROOT / cfg["paths"]["exp_dir"]
    tokenizer = load_tokenizer(cfg, exp_dir)
    model = build_model(cfg, tokenizer.vocab_size)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    model.eval()
    _disable_mha_fastpath()
    n_mha = rewrite_attention_for_export(model)
    log.info("rewrote %d MultiheadAttention modules for export", n_mha)
    wrapped = ExportWrapper(model.eval())
    feats = torch.randn(1, 300, int(cfg["model"].get("input_dim", 80)))
    lengths = torch.tensor([300], dtype=torch.long)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        (feats.clone(), lengths.clone()),
        str(out),
        input_names=["feats", "lengths"],
        output_names=["logits", "logit_lengths"],
        dynamic_axes={
            "feats": {0: "batch", 1: "time"},
            "lengths": {0: "batch"},
            "logits": {0: "batch", 1: "time"},
            "logit_lengths": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    log.info("wrote %s (%.1f MB)", out, out.stat().st_size / 1e6)
    # Reload weights for verify so tracing cannot poison the PyTorch graph.
    pt = build_model(cfg, tokenizer.vocab_size)
    pt.load_state_dict(blob["model"])
    pt.eval()
    _disable_mha_fastpath()
    rewrite_attention_for_export(pt)
    fresh = torch.randn(1, 240, int(cfg["model"].get("input_dim", 80)))
    _verify(ExportWrapper(pt), out, fresh, torch.tensor([240], dtype=torch.long))
    padded = torch.randn(2, 240, int(cfg["model"].get("input_dim", 80)))
    _verify(ExportWrapper(pt), out, padded, torch.tensor([240, 180], dtype=torch.long))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    cfg = load_yaml(args.config)
    out = args.out or ROOT / cfg["paths"]["exp_dir"] / "model.onnx"
    export(Path(args.config), Path(args.ckpt), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
