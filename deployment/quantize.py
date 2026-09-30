"""INT8 dynamic quantization (primary deployment target, Phase 17)."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import setup_logging

log = setup_logging()


def quantize_onnx(src: Path, dest: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    dest.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dest),
        weight_type=QuantType.QInt8,
        per_channel=True,
        extra_options={"EnableSubgraph": True},
    )
    src_mb = src.stat().st_size / 1e6
    dest_mb = dest.stat().st_size / 1e6
    log.info("wrote %s (%.1f MB, was %.1f MB, %.2fx)", dest, dest_mb, src_mb, src_mb / max(dest_mb, 1e-6))
    return dest


def quantize_torch(ckpt: Path, dest: Path) -> Path:
    import torch
    from torch.quantization import quantize_dynamic as tq

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    # Store weights only; full graph quantization is applied at load time in benchmark.py.
    torch.save(blob, dest)
    log.info("copied checkpoint to %s; apply torch.ao.quantization at runtime", dest)
    return dest


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    out = args.out or args.onnx.with_name(args.onnx.stem + ".int8.onnx")
    quantize_onnx(args.onnx, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
