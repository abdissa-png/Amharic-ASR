#!/usr/bin/env python3
"""Time one training step per autocast dtype and list the slowest CUDA ops.

Synthetic batches at the two shapes the 20 ms model sees: sentence-length
clips and 40 s clips. ``--vary N`` instead packs N batches of random clip
lengths up to the batch budget, so batch size and length change every step
the way they do in training. Run it with training stopped, or the two jobs
share the GPU.

    python bench_amp.py --config configs/50m_char_12layer_20ms.yaml
    python bench_amp.py --config configs/50m_char_12layer_20ms.yaml --profile
    python bench_amp.py --config configs/50m_char_12layer_20ms.yaml --vary 30
"""

from __future__ import annotations

import argparse
import random
import time

import torch

from models.ctc import build_model
from train import _make_adamw
from utils import ROOT, load_yaml

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
# (clips per batch, seconds per clip): 80 s of audio each.
SHAPES = {"sentence_8x10s": (8, 10.0), "long_2x40s": (2, 40.0)}


def make_batch(n: int, seconds: float, vocab: int, device: torch.device):
    t = int(seconds * 100)
    feats = torch.randn(n, t, 80, device=device)
    lens = torch.full((n,), t, dtype=torch.long, device=device)
    u = max(1, t // 8)
    targets = torch.randint(1, vocab, (n, u), device=device)
    tlens = torch.full((n,), u, dtype=torch.long, device=device)
    return feats, lens, targets, tlens


def varied_batches(count: int, budget_s: float, vocab: int, device: torch.device, bucket: int, seed: int = 0):
    """Pack random 1.5–20 s clips like DurationBatchSampler; one new shape per batch."""
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        durs: list[float] = []
        acc = 0.0
        while True:
            d = rng.uniform(1.5, 20.0)
            if durs and acc + d > budget_s:
                break
            durs.append(d)
            acc += d
        frames = [int(d * 100) for d in durs]
        t = max(frames)
        if bucket > 0:
            t = -(-t // bucket) * bucket
        n = len(frames)
        feats = torch.randn(n, t, 80, device=device)
        lens = torch.tensor(frames, dtype=torch.long, device=device)
        us = [max(1, f // 8) for f in frames]
        targets = torch.randint(1, vocab, (n, max(us)), device=device)
        tlens = torch.tensor(us, dtype=torch.long, device=device)
        out.append((feats, lens, targets, tlens))
    return out


def train_step(model, opt, scaler, batch, dtype, inter_weight: float) -> None:
    feats, lens, targets, tlens = batch
    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        loss, _, _ = model.loss(feats, lens, targets, tlens, inter_weight=inter_weight)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--dtypes", nargs="+", default=["bf16", "fp16"], choices=sorted(DTYPES))
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--vocab", type=int, default=347)
    p.add_argument("--cudnn-benchmark", action="store_true")
    p.add_argument("--no-cudnn", action="store_true", help="torch.backends.cudnn.enabled = False")
    p.add_argument("--vary", type=int, default=0, help="time N batches with a new shape each step")
    p.add_argument("--bucket", type=int, default=0, help="with --vary, round padded length up to this many frames")
    p.add_argument("--profile", action="store_true", help="print the slowest CUDA ops per dtype and shape")
    args = p.parse_args()

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    torch.backends.cudnn.enabled = not args.no_cudnn
    cfg = load_yaml(ROOT / args.config)
    tcfg = cfg["train"]
    inter = float(tcfg.get("inter_ctc_weight", 0.3))
    print(
        f"{torch.cuda.get_device_name(0)}  torch {torch.__version__}  "
        f"cudnn={'off' if args.no_cudnn else 'on'} cudnn.benchmark={args.cudnn_benchmark} "
        f"vary={args.vary} bucket={args.bucket}"
    )

    rows = []
    for name in args.dtypes:
        dtype = DTYPES[name]
        torch.manual_seed(0)
        model = build_model(cfg, args.vocab).to(device).train()
        opt = _make_adamw(model.parameters(), tcfg, device)
        scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
        if args.vary:
            budget = float(tcfg.get("batch_seconds", 80))
            batches = varied_batches(args.vary + args.warmup, budget, args.vocab, device, args.bucket)
            for batch in batches[: args.warmup]:
                train_step(model, opt, scaler, batch, dtype, inter)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for batch in batches[args.warmup :]:
                train_step(model, opt, scaler, batch, dtype, inter)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000 / args.vary
            peak = torch.cuda.max_memory_allocated() / 2**30
            rows.append((name, "varied", ms, peak))
            print(f"{name:5s} {'varied':15s} {ms:8.0f} ms/step  peak {peak:5.2f} GB", flush=True)
            del model, opt, scaler
            torch.cuda.empty_cache()
            continue
        for shape, (n, seconds) in SHAPES.items():
            batch = make_batch(n, seconds, args.vocab, device)
            for _ in range(args.warmup):
                train_step(model, opt, scaler, batch, dtype, inter)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for _ in range(args.steps):
                train_step(model, opt, scaler, batch, dtype, inter)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000 / args.steps
            peak = torch.cuda.max_memory_allocated() / 2**30
            rows.append((name, shape, ms, peak))
            print(f"{name:5s} {shape:15s} {ms:8.0f} ms/step  peak {peak:5.2f} GB", flush=True)

            if args.profile:
                from torch.profiler import ProfilerActivity, profile

                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    for _ in range(2):
                        train_step(model, opt, scaler, batch, dtype, inter)
                    torch.cuda.synchronize()
                print(f"\n--- {name} {shape}: top CUDA ops ---")
                print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
        del model, opt, scaler
        torch.cuda.empty_cache()

    print("\nsummary")
    for name, shape, ms, peak in rows:
        print(f"  {name:5s} {shape:15s} {ms:8.0f} ms/step  peak {peak:5.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
