#!/usr/bin/env python3
"""Find what keeps the Zipformer training GPU idle.

Run it on the training machine with training stopped. Three numbers decide
what to change:

* ``loader``: audio seconds per wall second from the train DataLoader alone
  (CPU features + waveform augmentation). It has to beat the step rate.
* ``step``: ``train_zipformer.train_step`` on cached real batches, so the
  loader is not in the timing. ``GPU busy`` is the fraction of wall time
  with a CUDA kernel running. Well under 100% with the loader excluded
  means the step is bound by Python and kernel launches.
* ``--syncs``: places in one step where the CPU waits for the GPU.

    python bench_zipformer.py --config configs/zipformer_s_streaming_pool.yaml loader
    python bench_zipformer.py --config configs/zipformer_s_streaming_pool.yaml step
    python bench_zipformer.py --config configs/zipformer_s_streaming_pool.yaml step --batch-seconds 120 --syncs

The step at a small ``--batch-seconds`` (10) is close to the fixed per-step
CPU cost, which does not grow with the batch. On a Quadro T2000 with an
i7-9850H that floor is about 0.65 s.
"""

from __future__ import annotations

import argparse
import collections
import time
import traceback
import warnings
from pathlib import Path

import torch

import train_zipformer as tz
from models.zipformer_icefall import build_zipformer, eden, load_icefall, scaled_adam
from train import load_cmvn, load_tokenizer, make_loader
from utils import ROOT, load_yaml, set_seed


def _cfg(args) -> dict:
    cfg = load_yaml(ROOT / args.config)
    tcfg = cfg["train"]
    if args.batch_seconds:
        tcfg["batch_seconds"] = args.batch_seconds
    if args.workers is not None:
        tcfg["num_workers"] = args.workers
    if args.no_augment:
        tcfg.pop("augment", None)
    if args.metadata_dir:
        cfg["paths"]["metadata_dir"] = args.metadata_dir
    return cfg


def _exp_dir(cfg: dict, args) -> Path:
    if args.exp_dir:
        return ROOT / args.exp_dir
    init = cfg.get("paths", {}).get("init_from")
    return (ROOT / init).parent if init else ROOT / cfg["paths"]["exp_dir"]


def bench_loader(cfg: dict, exp: Path, batches: int) -> None:
    tok = load_tokenizer(cfg, exp)
    loader = make_loader("train", tok, cfg, train=True, cmvn=load_cmvn(cfg, exp))
    t0 = time.perf_counter()
    audio, n = 0.0, 0
    for batch in loader:
        audio += sum(batch["duration_s"])
        n += 1
        if n >= batches:
            break
    dt = time.perf_counter() - t0
    tcfg = cfg["train"]
    print(
        f"loader: workers={tcfg.get('num_workers', 4)} augment={'augment' in tcfg} "
        f"{n} batches, {audio:.0f} s audio in {dt:.1f} s = {audio / dt:.0f}x realtime, "
        f"{dt / n:.2f} s/batch"
    )


def report_env() -> None:
    """Things that make every kernel launch slow, independent of the model."""
    import os
    import subprocess

    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    load = os.getloadavg() if hasattr(os, "getloadavg") else (float("nan"),) * 3
    print(
        f"env: cpus={cpus} load_avg_1m={load[0]:.1f} torch_threads={torch.get_num_threads()} "
        f"CUDA_LAUNCH_BLOCKING={os.environ.get('CUDA_LAUNCH_BLOCKING', '(unset)')}"
    )
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        others = [line for line in out.splitlines() if line and not line.startswith(f"{os.getpid()},")]
        print("  other GPU processes: " + ("; ".join(others) if others else "none"))
    except (OSError, subprocess.SubprocessError):
        print("  other GPU processes: nvidia-smi unavailable")
    x = torch.zeros(1, device="cuda")
    for _ in range(200):
        x.add_(1)
    torch.cuda.synchronize()
    n = 5000
    t0 = time.perf_counter()
    for _ in range(n):
        x.add_(1)
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / n * 1e6
    print(f"  tiny-kernel launch: {us:.1f} us each (i7-9850H + T2000: about 8 us)")
    print(f"  cgroup cpu quota: {_cpu_quota()}")


def _read(path: str) -> str:
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def _cpu_quota() -> str:
    v2 = _read("/sys/fs/cgroup/cpu.max").split()
    if len(v2) == 2:
        return "none" if v2[0] == "max" else f"{int(v2[0]) / int(v2[1]):.2f} cpus"
    quota = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").strip()
    period = _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us").strip()
    if quota and period:
        return "none" if int(quota) < 0 else f"{int(quota) / int(period):.2f} cpus"
    return "unknown"


def _cpu_counters() -> dict:
    c = {"wall": time.perf_counter(), "cpu": time.process_time()}
    tasks = [_read(str(p)).split() for p in Path("/proc/self/task").glob("*/schedstat")]
    tasks = [t for t in tasks if len(t) >= 2]
    if tasks:
        c["run"] = sum(int(t[0]) for t in tasks) / 1e9
        c["wait"] = sum(int(t[1]) for t in tasks) / 1e9
    for path, key, unit in (
        ("/sys/fs/cgroup/cpu.stat", "throttled_usec", 1e6),
        ("/sys/fs/cgroup/cpu/cpu.stat", "throttled_time", 1e9),
    ):
        for line in _read(path).splitlines():
            k, _, v = line.partition(" ")
            if k == key:
                c["throttled"] = int(v) / unit
            elif k == "nr_throttled":
                c["nr_throttled"] = int(v)
    stat = _read("/proc/stat").splitlines()
    if stat and stat[0].startswith("cpu "):
        f = [int(x) for x in stat[0].split()[1:9]]
        c["steal"], c["total"] = (f[7] if len(f) > 7 else 0), sum(f)
    return c


def _cpu_report(a: dict, b: dict) -> str:
    wall = b["wall"] - a["wall"]
    parts = [f"process cpu {100 * (b['cpu'] - a['cpu']) / wall:.0f}% of wall (all threads)"]
    if "run" in a:
        parts.append(
            f"threads waiting for a cpu {100 * (b['wait'] - a['wait']) / wall:.0f}% of wall "
            f"(running {100 * (b['run'] - a['run']) / wall:.0f}%)"
        )
    if "throttled" in a:
        parts.append(
            f"cgroup throttled {b['throttled'] - a['throttled']:.1f} s "
            f"({b.get('nr_throttled', 0) - a.get('nr_throttled', 0)} times)"
        )
    if "steal" in a and b["total"] > a["total"]:
        parts.append(f"vm steal {100 * (b['steal'] - a['steal']) / (b['total'] - a['total']):.0f}%")
    return "  cpu while timing: " + "; ".join(parts)


def _gpu_busy(prof) -> float:
    spans = sorted(
        (e.time_range.start, e.time_range.end)
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA
    )
    busy, cur = 0.0, None
    for s, e in spans:
        if cur is None or s > cur[1]:
            if cur:
                busy += cur[1] - cur[0]
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    if cur:
        busy += cur[1] - cur[0]
    return busy / 1e6


def _count_syncs(run) -> collections.Counter:
    counts: collections.Counter = collections.Counter()

    def hook(message, category, filename, lineno, file=None, line=None):
        if "synchroniz" not in str(message):
            return
        frames = [
            f
            for f in traceback.extract_stack()[:-1]
            if str(ROOT) in f.filename or "k2" in f.filename
        ]
        frames = [f for f in frames if not f.filename.endswith("bench_zipformer.py")]
        counts[" <- ".join(f"{Path(f.filename).name}:{f.lineno}" for f in frames[-3:][::-1])] += 1

    shown, warnings.showwarning = warnings.showwarning, hook
    warnings.simplefilter("always")
    torch.cuda.set_sync_debug_mode("warn")
    try:
        run()
    finally:
        torch.cuda.set_sync_debug_mode(0)
        warnings.showwarning = shown
    return counts


def bench_step(cfg: dict, exp: Path, args) -> None:
    load_icefall()
    from icefall.utils import create_grad_scaler

    set_seed(0)
    tcfg = cfg["train"]
    tcfg.pop("augment", None)
    tok = load_tokenizer(cfg, exp)
    loader = tz._rebuild(make_loader("train", tok, cfg, train=True, cmvn=load_cmvn(cfg, exp)), cfg, True)
    keys = ("feats", "feat_lens", "targets", "target_lens")
    batches = []
    for b in loader:
        batches.append({k: b[k].pin_memory() for k in keys})
        if len(batches) >= args.steps + args.warmup:
            break
    del loader

    device = torch.device("cuda")
    report_env()
    torch.backends.cudnn.enabled = bool(tcfg.get("cudnn", False))
    torch.backends.cudnn.benchmark = bool(tcfg.get("cudnn_benchmark", False))
    model = build_zipformer(cfg, tok.vocab_size)
    ckpt = exp / "best.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    model.to(device).train()
    opt = scaled_adam(model, float(tcfg["base_lr"]))
    sched = eden(opt, tcfg)
    use_amp, amp_dtype = tz._amp_setup(tcfg, device)
    scaler = create_grad_scaler("cuda", enabled=use_amp and amp_dtype == torch.float16, init_scale=1.0)
    batch_seconds = float(tcfg.get("batch_seconds", 80))
    ref = float(tcfg.get("ref_duration", 600))
    idx = [0]

    def step(batch):
        tz.set_batch_count(model, tz.adjusted_batch_count(idx[0], batch_seconds, ref))
        idx[0] += 1
        tz.train_step(
            model, opt, sched, scaler, batch, device, tcfg, idx[0],
            epoch=10**6, amp_dtype=amp_dtype, use_amp=use_amp,
        )

    for b in batches[: args.warmup]:
        step(b)
    torch.cuda.synchronize()
    timed = batches[args.warmup :]
    audio = sum(float(b["feat_lens"].sum()) / 100 for b in timed)

    from torch.profiler import ProfilerActivity, profile

    c0 = _cpu_counters()
    t0 = time.perf_counter()
    for b in timed:
        step(b)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    c1 = _cpu_counters()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        p0 = time.perf_counter()
        for b in timed[: min(4, len(timed))]:
            step(b)
        torch.cuda.synchronize()
        p_wall = time.perf_counter() - p0
    print(
        f"step: {torch.cuda.get_device_name(0)} batch_seconds={batch_seconds:.0f} "
        f"precision={tcfg.get('precision')} cudnn={torch.backends.cudnn.enabled}\n"
        f"  {wall / len(timed) * 1000:.0f} ms/step, {audio / wall:.0f}x realtime, "
        f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB\n"
        f"  GPU busy {100 * _gpu_busy(prof) / p_wall:.0f}% of the step (profiled steps)\n"
        + _cpu_report(c0, c1)
    )
    if args.top:
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=args.top))
    if args.syncs:
        counts = _count_syncs(lambda: step(timed[0]))
        print(f"  syncs in one step: {sum(counts.values())}")
        for where, n in counts.most_common(15):
            print(f"    {n:3d}  {where or '(torch internals)'}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("mode", choices=["loader", "step"])
    p.add_argument("--exp-dir", help="tokenizer, cmvn.pt and best.pt; default is init_from's folder")
    p.add_argument("--metadata-dir", help="folder with train.jsonl; default is the config's")
    p.add_argument("--batch-seconds", type=float)
    p.add_argument("--workers", type=int)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--batches", type=int, default=40, help="loader mode")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--syncs", action="store_true")
    p.add_argument("--top", type=int, default=0, help="print the N ops with the most CPU time")
    args = p.parse_args()
    cfg = _cfg(args)
    exp = _exp_dir(cfg, args)
    if args.mode == "loader":
        bench_loader(cfg, exp, args.batches)
    else:
        bench_step(cfg, exp, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
