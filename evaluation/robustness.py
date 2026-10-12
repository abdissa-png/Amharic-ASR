"""Score a checkpoint under acoustic conditions and dialect slices.

Train-time randomization lives in ``data.augment``. This module is the
eval counterpart: apply one condition at a time, and also bucket Dataset.ET
rows by ``region`` / ``corpus`` so a single number cannot hide a dialect.

    python -m evaluation.robustness --config configs/20m_char_12layer.yaml \\
        --ckpt exp/20m_char_12layer/best.pt --split dev \\
        --conditions clean,noisy,reverb,codec,gain_low

    python -m evaluation.robustness --config configs/20m_char_12layer.yaml \\
        --ckpt exp/20m_char_12layer/best.pt --split dev --by region
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data.dataset import AsrDataset, collate
from data.oov import vocab_from_rows
from data.prepare import load_split_jsonl, metadata_dir
from decoding.greedy import greedy_decode
from evaluation.metrics import score_split
from models.features import SAMPLE_RATE
from train import load_cmvn, load_tokenizer
from utils import ROOT, load_jsonl, load_yaml, save_json, setup_logging

log = setup_logging()

# Acoustic conditions actually implemented as waveform transforms.
ACOUSTIC = (
    "clean",
    "noisy",
    "reverb",
    "gain_low",
    "gain_et",
    "gain_high",
    "gain_rand",
    "codec",
)
# Metadata buckets — no waveform change.
BUCKET_FIELDS = ("region", "corpus")

# Kept for the original Phase-20 list; unimplemented names raise in apply_condition.
CATEGORIES = ACOUSTIC + (
    "device_mismatch",
    "fast_speech",
    "spontaneous",
    "dialect",
    "names",
    "numbers",
    "rare_words",
    "morphologically_complex",
)


def score_categories(
    grouped: dict[str, tuple[list[dict[str, Any]], list[str]]],
    *,
    v_train: set[str] | None = None,
) -> dict[str, Any]:
    return {name: score_split(rows, hyps, v_train=v_train) for name, (rows, hyps) in grouped.items()}


def group_by_field(rows: list[dict[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get(field) or "").strip() or "(not stated)"
        grouped[key].append(row)
    return dict(grouped)


def identity_loader(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """No-op, kept so older imports do not break."""
    return rows


def _decode_loader(model, loader, tokenizer, device) -> tuple[list[str], list[str]]:
    hyps: list[str] = []
    clip_ids: list[str] = []
    model_eval = getattr(model, "eval", None)
    if callable(model_eval):
        model.eval()
    for batch in loader:
        feats = batch["feats"].to(device)
        lens = batch["feat_lens"].to(device)
        with torch.inference_mode():
            logits, logit_lens = model(feats, lens)
        hyps.extend(greedy_decode(logits, tokenizer, logit_lens))
        clip_ids.extend(batch["clip_id"])
    return clip_ids, hyps


def _ordered_hyps(rows: list[dict[str, Any]], clip_ids: list[str], hyps: list[str]) -> list[str]:
    hyp_map = dict(zip(clip_ids, hyps))
    return [hyp_map.get(r["clip_id"], "") for r in rows]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    p.add_argument("--jsonl", type=Path, default=None)
    p.add_argument("--train-jsonl", type=Path, default=None)
    p.add_argument(
        "--conditions",
        default="clean,noisy,reverb,codec,gain_low",
        help="Comma-separated acoustic conditions from: " + ",".join(ACOUSTIC),
    )
    p.add_argument("--by", default=None, choices=list(BUCKET_FIELDS), help="Also score metadata buckets.")
    p.add_argument("--limit", type=int, default=None, help="Score only the first N clips (smoke test).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--noise-dir", type=Path, default=ROOT / "data" / "noise")
    p.add_argument("--rir-dir", type=Path, default=ROOT / "data" / "rir")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    exp_dir = ROOT / cfg["paths"]["exp_dir"]
    jsonl = Path(args.jsonl) if args.jsonl else metadata_dir(cfg) / f"{args.split}.jsonl"
    if not jsonl.is_absolute():
        jsonl = ROOT / jsonl
    out_dir = args.out_dir or (exp_dir / f"eval_robust_{jsonl.stem}")
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from evaluate import load_model

    tokenizer = load_tokenizer(cfg, exp_dir)
    cmvn = load_cmvn(cfg, exp_dir, compute_if_missing=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, Path(args.ckpt), tokenizer, device)

    rows = load_jsonl(jsonl)
    if args.limit:
        rows = rows[: args.limit]
        log.info("limit=%d", len(rows))

    v_train = None
    if args.train_jsonl:
        train_path = args.train_jsonl if args.train_jsonl.is_absolute() else ROOT / args.train_jsonl
        v_train = set(vocab_from_rows(load_jsonl(train_path)))
    else:
        train_path = metadata_dir(cfg) / "train.jsonl"
        if train_path.exists():
            v_train = set(vocab_from_rows(load_split_jsonl("train", metadata_dir(cfg))))

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in conditions if c not in ACOUSTIC]
    if unknown:
        p.error(f"unknown conditions {unknown}; choose from {list(ACOUSTIC)}")

    # Temporary jsonl so AsrDataset can take a limit subset.
    subset_path = out_dir / "_subset.jsonl"
    from utils import write_jsonl

    write_jsonl(subset_path, rows)

    result: dict[str, Any] = {
        "jsonl": str(jsonl),
        "split": jsonl.stem,
        "n": len(rows),
        "sample_rate": SAMPLE_RATE,
        "conditions": {},
        "vs_clean": {},
    }
    hyp_by_cond: dict[str, list[str]] = {}

    for cond in conditions:
        log.info("condition=%s", cond)
        ds = AsrDataset(
            subset_path,
            tokenizer,
            train=False,
            cmvn=cmvn,
            robustness=None if cond == "clean" else cond,
            noise_dir=args.noise_dir,
            rir_dir=args.rir_dir,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
        clip_ids, hyps = _decode_loader(model, loader, tokenizer, device)
        ordered = _ordered_hyps(rows, clip_ids, hyps)
        hyp_by_cond[cond] = ordered
        scores = score_split(rows, ordered, v_train=v_train)
        result["conditions"][cond] = scores
        write_jsonl(
            out_dir / f"hypotheses_{cond}.jsonl",
            [{"clip_id": r["clip_id"], "text": h} for r, h in zip(rows, ordered)],
        )
        log.info(
            "  %s  WER=%.2f CER=%.2f",
            cond,
            scores["overall"]["wer_pct"],
            scores["overall"]["cer_pct"],
        )

    if "clean" in hyp_by_cond:
        clean = hyp_by_cond["clean"]
        samples: list[dict[str, Any]] = []
        for cond, hyps in hyp_by_cond.items():
            if cond == "clean":
                continue
            n_chg = sum(a != b for a, b in zip(clean, hyps))
            result["vs_clean"][cond] = {
                "n_changed": n_chg,
                "frac_changed": round(n_chg / max(1, len(clean)), 4),
            }
        for i, row in enumerate(rows):
            variants = {c: hyp_by_cond[c][i] for c in hyp_by_cond}
            if len(set(variants.values())) <= 1:
                continue
            samples.append(
                {
                    "clip_id": row["clip_id"],
                    "ref": row.get("text_raw") or row.get("text"),
                    **variants,
                }
            )
            if len(samples) >= 8:
                break
        result["samples"] = samples

    if args.by:
        log.info("bucket by %s (clean audio)", args.by)
        groups = group_by_field(rows, args.by)
        ds = AsrDataset(subset_path, tokenizer, train=False, cmvn=cmvn)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
        clip_ids, hyps = _decode_loader(model, loader, tokenizer, device)
        hyp_map = dict(zip(clip_ids, hyps))
        buckets: dict[str, Any] = {}
        for key, group in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            g_hyps = [hyp_map.get(r["clip_id"], "") for r in group]
            buckets[key] = score_split(group, g_hyps, v_train=v_train)
            buckets[key]["n_utt"] = len(group)
            log.info(
                "  %s=%s n=%d WER=%.2f CER=%.2f",
                args.by,
                key,
                len(group),
                buckets[key]["overall"]["wer_pct"],
                buckets[key]["overall"]["cer_pct"],
            )
        result["by"] = {args.by: buckets}

    save_json(out_dir / "robustness.json", result)
    log.info("wrote %s", out_dir / "robustness.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
