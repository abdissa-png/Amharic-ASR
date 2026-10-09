#!/usr/bin/env python3
"""Copy a running train_zipformer.py onto disk without another GPU allocation.

The epoch file is written only after the dev pass. This pauses the job,
copies each tensor to CPU, writes ``emergency.pt``, and lets training
continue. The GPU-resident weights stay where they are.

Run it on the studio, in another shell, while the step is still on the GPU:

    python save_live_zipformer.py
    python save_live_zipformer.py --pid 12345 --out exp/zipformer_s_streaming_pool_leyu/emergency.pt

Resume that file with the same config:

    python train_zipformer.py \\
        --config configs/zipformer_s_streaming_pool_leyu.yaml \\
        --resume exp/zipformer_s_streaming_pool_leyu/emergency.pt

``touch exp/.../SAVE_NOW`` or ``kill -USR1 <pid>`` only works for a process
started after the training loop learned those signals. A job that is already
running needs this script.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path


def to_cpu(obj):
    import torch

    if torch.is_tensor(obj):
        if obj.device.type == "cpu":
            return obj.detach().clone()
        return obj.detach().to("cpu")
    if isinstance(obj, dict):
        return {str(k): to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_cpu(v) for v in obj)
    return obj


def find_train_locals():
    """Locate the train() frame. Safe to call inside that process."""
    import sys

    found = None
    best = -1
    for frame in sys._current_frames().values():
        current = frame
        while current is not None:
            loc = current.f_locals
            model = loc.get("model")
            if model is not None and "opt" in loc and "batch_idx" in loc and "epoch" in loc:
                try:
                    import torch

                    n = sum(p.numel() for p in model.parameters()) if isinstance(model, torch.nn.Module) else 0
                except Exception:
                    n = 0
                if n >= best:
                    found = loc
                    best = n
            current = current.f_back
    return found


def dump_live(out_path: str) -> str:
    """Runs inside the training process. Returns 'ok' or 'err'."""
    import traceback
    from pathlib import Path

    import torch

    out = Path(out_path)
    log_path = out.with_suffix(".log")
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("started\n", encoding="utf-8")
    try:
        loc = find_train_locals()
        if loc is None:
            raise RuntimeError("train() frame not found; weights were not read")
        model = loc["model"]
        opt = loc["opt"]
        scheduler = loc["scheduler"]
        scaler = loc.get("scaler")
        payload = {
            "epoch": int(loc["epoch"]),
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else {},
            "config": loc.get("cfg"),
            "vocab_size": int(loc["tokenizer"].vocab_size) if loc.get("tokenizer") is not None else None,
            "batch_idx": int(loc["batch_idx"]),
            "best_wer": float(loc.get("best_wer") or 1e9),
            "best_ctc_wer": float(loc.get("best_ctc_wer") or 1e9),
            "bad": int(loc.get("bad") or 0),
            "history": loc.get("history") or [],
            "emergency": True,
        }
        copied = to_cpu(payload)
        tmp = out.with_suffix(out.suffix + ".tmp")
        torch.save(copied, tmp)
        tmp.replace(out)
        n = sum(p.numel() for p in model.parameters())
        log_path.write_text(
            f"ok\nparameters={n}\nepoch={payload['epoch']}\nbatch_idx={payload['batch_idx']}\n",
            encoding="utf-8",
        )
        print(
            f"emergency save {out} epoch={payload['epoch']} step={payload['batch_idx']} params={n}",
            flush=True,
        )
        return "ok"
    except Exception:
        log_path.write_text("err\n" + traceback.format_exc(), encoding="utf-8")
        return "err"


def _training_pids() -> list[int]:
    me = os.getpid()
    found: list[int] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return found
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == me:
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if "train_zipformer.py" in cmd:
            found.append(pid)
    return found


def _inject(pid: int, payload: Path, sudo: bool, gdb_log: Path) -> subprocess.Popen[str]:
    code = f"exec(open({str(payload)!r}, encoding='utf-8').read())"
    cmd = [
        "gdb",
        "-p",
        str(pid),
        "-batch",
        "-ex",
        "set pagination off",
        "-ex",
        "set confirm off",
        "-ex",
        "set unwindonsignal on",
        "-ex",
        "set $gil = (int)PyGILState_Ensure()",
        "-ex",
        f"call (int)PyRun_SimpleString({code!r})",
        "-ex",
        "call (void)PyGILState_Release($gil)",
        "-ex",
        "detach",
        "-ex",
        "quit",
    ]
    if sudo:
        cmd.insert(0, "sudo")
    handle = gdb_log.open("w", encoding="utf-8")
    return subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True)


def _resume(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGCONT)
    except OSError:
        pass


def save_pid(pid: int, out: Path, *, sudo: bool = False) -> int:
    out = out.resolve()
    log_path = out.with_suffix(".log")
    log_path.unlink(missing_ok=True)
    payload = out.with_suffix(".inject.py")
    payload.write_text(
        textwrap.dedent(
            f"""
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "save_live_zipformer", {str(Path(__file__).resolve())!r}
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.dump_live({str(out)!r})
            """
        ).lstrip(),
        encoding="utf-8",
    )
    print(f"pausing pid {pid} to copy weights to CPU", flush=True)
    gdb_log = out.with_suffix(".gdb.txt")
    proc = _inject(pid, payload, sudo, gdb_log)
    deadline = time.time() + 30
    started = False
    try:
        while True:
            if log_path.exists():
                text = log_path.read_text(encoding="utf-8", errors="replace")
                if text.startswith("started"):
                    started = True
                    deadline = max(deadline, time.time() + 600)
                if text.startswith("ok") or text.startswith("err"):
                    try:
                        proc.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break
            if proc.poll() is not None and not started:
                break
            if time.time() > deadline:
                break
            time.sleep(0.5)
    finally:
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        _resume(pid)
    gdb_out = gdb_log.read_text(encoding="utf-8", errors="replace") if gdb_log.exists() else ""
    if log_path.exists() and log_path.read_text(encoding="utf-8", errors="replace").startswith("ok"):
        print(log_path.read_text(encoding="utf-8"), end="")
        print(f"wrote {out}", flush=True)
        return 0
    if not started and gdb_out and (
        "Operation not permitted" in gdb_out or "ptrace" in gdb_out.lower()
    ):
        if not sudo:
            print("ptrace denied; retrying with sudo", flush=True)
            return save_pid(pid, out, sudo=True)
    sys.stderr.write(gdb_out)
    if log_path.exists():
        sys.stderr.write(log_path.read_text(encoding="utf-8", errors="replace"))
    else:
        sys.stderr.write(
            "The training process did not start the copy. "
            "If gdb hung on the GIL, the job has been continued and was not modified.\n"
        )
    return 1


def main() -> int:
    p = argparse.ArgumentParser(description="Save the running Zipformer to disk via CPU copies")
    p.add_argument("--pid", type=int, default=None)
    p.add_argument(
        "--out",
        default="exp/zipformer_s_streaming_pool_leyu/emergency.pt",
        help="checkpoint path written by the training process",
    )
    args = p.parse_args()
    if args.pid is None:
        pids = _training_pids()
        if len(pids) != 1:
            raise SystemExit(
                "expected one train_zipformer.py process, found "
                + (", ".join(str(pid) for pid in pids) or "none")
                + ". Pass --pid."
            )
        args.pid = pids[0]
    if subprocess.run(["gdb", "--version"], check=False, capture_output=True).returncode != 0:
        raise SystemExit("gdb is not installed. On the studio: sudo apt-get install -y gdb")
    return save_pid(args.pid, Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
