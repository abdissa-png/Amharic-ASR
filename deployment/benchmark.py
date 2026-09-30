"""CPU RTF / RAM benchmark for ONNX or PyTorch checkpoints.

    python -m deployment.benchmark --onnx exp/20m_char/model.onnx --split test --limit 50
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from data.prepare import load_split_jsonl
from evaluation.metrics import peak_rss_mb
from models.features import log_mel
from utils import ROOT, save_json, setup_logging

log = setup_logging()


def load_wav(path: Path) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(0)
    if sr != 16000:
        raise ValueError(sr)
    return audio


def benchmark_onnx(onnx_path: Path, rows: list[dict]) -> dict:
    import onnxruntime as ort
    import torch

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    lat = []
    audio_s = 0.0
    rss0 = peak_rss_mb()
    t0 = time.perf_counter()
    for row in rows:
        wav = load_wav(ROOT / row["audio"])
        audio_s += row["duration_s"]
        feats = log_mel(torch.from_numpy(wav)).unsqueeze(0).numpy()
        lengths = np.array([feats.shape[1]], dtype=np.int64)
        s = time.perf_counter()
        sess.run(None, {"feats": feats, "lengths": lengths})
        lat.append(time.perf_counter() - s)
    wall = time.perf_counter() - t0
    lat.sort()
    n = max(1, len(lat))
    return {
        "n": len(rows),
        "rtf": wall / audio_s if audio_s else float("nan"),
        "latency_mean_s": sum(lat) / n,
        "latency_p50_s": lat[n // 2],
        "latency_p95_s": lat[min(n - 1, int(0.95 * n))],
        "ram_mb_start": rss0,
        "ram_mb_peak": peak_rss_mb(),
        "backend": "onnxruntime-cpu",
        "model": str(onnx_path),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    rows = load_split_jsonl(args.split)[: args.limit]
    result = benchmark_onnx(args.onnx, rows)
    out = args.out or args.onnx.with_suffix(".bench.json")
    save_json(out, result)
    log.info("RTF=%.3f p50=%.3fs RAM=%.0fMB", result["rtf"], result["latency_p50_s"], result["ram_mb_peak"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
