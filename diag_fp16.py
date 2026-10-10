#!/usr/bin/env python3
"""Find where an fp16 training step overflows, on real batches and a real checkpoint.

Runs forward and backward under fp16 autocast with no loss scaling, so any
inf comes from the weights and data, not from the scaler. Reports fp16
matmul outputs that hit inf (attention scores are the usual suspect: a NaN
from softmax(inf) is zeroed by nan_to_num, so the loss still looks fine),
the largest activation per module, and parameters with non-finite grads.
Stop training first; this needs the GPU.

    python diag_fp16.py --config configs/50m_char_12layer_20ms.yaml --batches 40
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import torch

from models.ctc import build_model
from train import load_cmvn, load_tokenizer, make_loader
from utils import ROOT, load_yaml


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", default=None, help="default: <exp_dir>/last.pt")
    p.add_argument("--batches", type=int, default=40)
    args = p.parse_args()

    torch.backends.cudnn.enabled = False
    device = torch.device("cuda")
    cfg = load_yaml(ROOT / args.config)
    cfg["train"]["num_workers"] = 2
    exp_dir = ROOT / cfg["paths"]["exp_dir"]
    tokenizer = load_tokenizer(cfg, exp_dir)
    model = build_model(cfg, tokenizer.vocab_size).to(device).train()
    ckpt_path = args.ckpt or exp_dir / "last.pt"
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    print(f"checkpoint {ckpt_path}  epoch {blob.get('epoch')}  scaler {blob.get('scaler')}")
    loader = make_loader("train", tokenizer, cfg, train=True, cmvn=load_cmvn(cfg, exp_dir))

    act_max: dict[str, float] = defaultdict(float)
    act_inf: Counter = Counter()

    def hook(name):
        def f(_mod, _inp, out):
            o = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(o) and o.is_floating_point():
                o = o.detach()
                finite = o[torch.isfinite(o)]
                if finite.numel():
                    act_max[name] = max(act_max[name], float(finite.abs().max()))
                if torch.isinf(o).any():
                    act_inf[name] += 1
        return f

    for name, mod in model.named_modules():
        if name:
            mod.register_forward_hook(hook(name))

    real_matmul = torch.matmul
    mm_max = [0.0]
    mm_inf = Counter()

    def matmul(a, b, *rest, **kw):
        out = real_matmul(a, b, *rest, **kw)
        if out.dtype == torch.float16:
            if torch.isinf(out).any():
                mm_inf[tuple(out.shape[-2:])] += 1
            finite = out[torch.isfinite(out)]
            if finite.numel():
                mm_max[0] = max(mm_max[0], float(finite.abs().max()))
        return out

    torch.matmul = matmul
    bad_steps = 0
    bad_params: Counter = Counter()
    fwd_inf_steps = 0
    try:
        for i, batch in enumerate(loader, 1):
            if i > args.batches:
                break
            before = sum(mm_inf.values()) + sum(act_inf.values())
            feats = batch["feats"].to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss, _, _ = model.loss(
                    feats,
                    batch["feat_lens"].to(device),
                    batch["targets"].to(device),
                    batch["target_lens"].to(device),
                    inter_weight=float(cfg["train"].get("inter_ctc_weight", 0.3)),
                )
            loss.backward()
            if sum(mm_inf.values()) + sum(act_inf.values()) > before:
                fwd_inf_steps += 1
            bad = [n for n, q in model.named_parameters() if q.grad is not None and not torch.isfinite(q.grad).all()]
            if bad:
                bad_steps += 1
                bad_params.update(bad)
            gmax = max(float(q.grad[torch.isfinite(q.grad)].abs().max()) for q in model.parameters() if q.grad is not None and torch.isfinite(q.grad).any())
            print(
                f"batch {i:3d}  B={feats.size(0):2d} T={feats.size(1):5d}  loss {float(loss):7.3f}  "
                f"max|grad| {gmax:9.3g}  nonfinite-grad params {len(bad):3d}",
                flush=True,
            )
            model.zero_grad(set_to_none=True)
    finally:
        torch.matmul = real_matmul

    n = min(args.batches, i)
    print(f"\n{bad_steps}/{n} batches had non-finite grads at loss scale 1")
    print(f"{fwd_inf_steps}/{n} batches had an inf somewhere in the fp16 forward")
    print(f"largest finite fp16 matmul output: {mm_max[0]:.0f}  (fp16 max 65504)")
    if mm_inf:
        print("fp16 matmuls that overflowed (output shape -> count):", dict(mm_inf.most_common(8)))
    if act_inf:
        print("modules whose output had inf:", act_inf.most_common(10))
    print("largest activations:")
    for name, v in sorted(act_max.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {v:10.1f}  {name}")
    if bad_params:
        print("params most often non-finite:", bad_params.most_common(10))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
