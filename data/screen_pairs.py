"""Flag clips whose audio does not match their transcript.

Dataset.ET test has clips where the audio is cut mid-sentence or belongs to
another sentence: 13 words in 3.6 s, decoded as the first four words or as an
unrelated sentence. CTC can still fit those pairs, so ``zero_infinity`` does
not drop them and training learns to invent words. Three checks:

* ``fast``      — more than ``--max-cps`` characters (no spaces) per second.
                  Clean Dataset.ET short clips sit at ≤ 11.3; the bad ones at 13.6–17.
* ``short_hyp`` — greedy CER above ``--short-hyp-cer`` and the hypothesis less
                  than ``--min-len-ratio`` of the reference length. A hard but
                  correct clip (dialect, noise) yields a wrong hypothesis of about
                  the right length; a mismatched pair yields a short one.
* ``garbage``   — greedy CER above ``--max-cer``.

The model check uses a checkpoint trained on these same clips, so it can have
memorised some bad pairs; the rate check does not depend on the model.

Every clip's stats go to ``screen_<split>.jsonl``, so thresholds can be changed
with ``--reuse`` without decoding again. Kept clips also get ``word_times``,
per-word CTC alignments, which training uses to cut 1–3 word crops. ``--out-dir``
gets a filtered copy of each screened split and an unchanged copy of every other
jsonl in the source folder, so a config can point ``paths.metadata_dir`` at it.

    python -m data.screen_pairs --config configs/20m_char_12layer_pool.yaml \\
        --ckpt exp/20m_char_12layer_pool/best.pt \\
        --metadata-dir metadata/pool --splits train --out-dir metadata/pool_clean
    python -m data.screen_pairs ... --reuse --max-cps 14
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import jiwer
import torch
from torch.utils.data import DataLoader

from data.dataset import AsrDataset, collate
from decoding.greedy import greedy_decode
from evaluation.geez_eval import normalize as geez_normalize
from models.ctc import build_model
from train import load_cmvn, load_tokenizer
from utils import ROOT, load_jsonl, load_yaml, save_json, setup_logging, write_jsonl

log = setup_logging()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def chars_per_second(row: dict) -> float:
    dur = float(row.get("duration_s") or 0.0)
    n = len((row.get("text") or "").replace(" ", ""))
    return n / dur if dur > 0 else float("inf")


def _sorted_batches(rows: list[dict], max_seconds: float) -> list[list[int]]:
    """Duration-sorted batches: little padding, so decoding is fast."""
    order = sorted(range(len(rows)), key=lambda i: float(rows[i].get("duration_s") or 0.0))
    batches: list[list[int]] = []
    cur: list[int] = []
    longest = 0.0
    for i in order:
        d = max(0.5, float(rows[i].get("duration_s") or 0.0))
        if cur and max(longest, d) * (len(cur) + 1) > max_seconds:
            batches.append(cur)
            cur, longest = [], 0.0
        cur.append(i)
        longest = max(longest, d)
    if cur:
        batches.append(cur)
    return batches


def word_times(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    space_id: int,
    frame_s: float,
    n_words: int,
) -> list[list[float]] | None:
    """``[start_s, end_s]`` per word from a CTC forced alignment of the reference.

    ``log_probs`` is (T, V) for one clip. Times mark the frames where the model
    places each word's characters; CTC fires on a character near its onset, so
    a word's sound can run a little past its ``end_s``. None if the reference
    cannot be aligned or its words do not match ``n_words``.
    """
    import torchaudio.functional as AF

    if targets.numel() == 0 or n_words == 0:
        return None
    try:
        labels, scores = AF.forced_align(
            log_probs[None].float().cpu(), targets[None].to(torch.int32).cpu(), blank=0
        )
    except RuntimeError:
        return None
    spans = AF.merge_tokens(labels[0], scores[0].exp(), blank=0)
    if len(spans) != targets.numel():
        return None
    out: list[list[float]] = []
    start = end = None
    for tok, span in zip(targets.tolist(), spans):
        if tok == space_id:
            if start is not None:
                out.append([round(start * frame_s, 3), round(end * frame_s, 3)])
            start = None
            continue
        if start is None:
            start = span.start
        end = span.end
    if start is not None:
        out.append([round(start * frame_s, 3), round(end * frame_s, 3)])
    return out if len(out) == n_words else None


def decode_split(
    rows: list[dict], jsonl: Path, cfg: dict, ckpt: Path, batch_seconds: float
) -> tuple[list[str], list[list[list[float]] | None]]:
    exp_dir = ROOT / cfg["paths"]["exp_dir"]
    tokenizer = load_tokenizer(cfg, exp_dir)
    cmvn = load_cmvn(cfg, exp_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, tokenizer.vocab_size)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
    model.to(device).eval()
    frame_s = 0.01 * int(model.encoder.time_ratio)
    space_id = tokenizer.stoi.get(" ", -1) if hasattr(tokenizer, "stoi") else -1

    ds = AsrDataset(jsonl, tokenizer, train=False, cmvn=cmvn)
    loader = DataLoader(
        ds,
        batch_sampler=_sorted_batches(ds.rows, batch_seconds),
        collate_fn=collate,
        num_workers=4,
    )
    hyps: dict[str, str] = {}
    times: dict[str, list[list[float]] | None] = {}
    for n, batch in enumerate(loader, 1):
        with torch.inference_mode():
            logits, lens = model(batch["feats"].to(device), batch["feat_lens"].to(device))
        hyps.update(zip(batch["clip_id"], greedy_decode(logits, tokenizer, lens)))
        if space_id >= 0:
            log_probs = logits.float().log_softmax(-1)
            for i, cid in enumerate(batch["clip_id"]):
                ids = batch["targets"][i, : int(batch["target_lens"][i])]
                n_words = len(batch["text"][i].split())
                times[cid] = word_times(log_probs[i, : int(lens[i])], ids, space_id, frame_s, n_words)
        if n % 200 == 0:
            log.info("decoded %d/%d clips", len(hyps), len(rows))
    return [hyps.get(r["clip_id"], "") for r in rows], [times.get(r["clip_id"]) for r in rows]


def clip_stats(row: dict, hyp: str, times: list[list[float]] | None = None) -> dict:
    ref = geez_normalize(row.get("text") or "", fold_geez=True)
    h = geez_normalize(hyp, fold_geez=True)
    ref_chars = len(ref.replace(" ", ""))
    cer = float(jiwer.cer(ref, h)) if ref.strip() else 0.0
    return {
        "clip_id": row["clip_id"],
        "corpus": row.get("corpus", ""),
        "duration_s": float(row.get("duration_s") or 0.0),
        "cps": round(chars_per_second(row), 2),
        "cer": round(cer, 4),
        "len_ratio": round(len(h.replace(" ", "")) / max(1, ref_chars), 3),
        "ref": row.get("text", ""),
        "hyp": hyp,
        "word_times": times,
    }


def flags_for(s: dict, args) -> list[str]:
    out = []
    if s["cps"] > args.max_cps:
        out.append("fast")
    if s["cer"] > args.short_hyp_cer and s["len_ratio"] < args.min_len_ratio:
        out.append("short_hyp")
    if s["cer"] > args.max_cer:
        out.append("garbage")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Config of the screening checkpoint (tokenizer, CMVN, model).")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--metadata-dir", type=Path, default=Path("metadata/pool"))
    p.add_argument("--splits", nargs="+", default=["train"])
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--reuse", action="store_true", help="Re-threshold existing screen_<split>.jsonl; no decoding.")
    p.add_argument("--batch-seconds", type=float, default=600.0, help="Padded audio seconds per decode batch.")
    p.add_argument("--max-cps", type=float, default=13.0)
    p.add_argument("--short-hyp-cer", type=float, default=0.5)
    p.add_argument("--min-len-ratio", type=float, default=0.5)
    p.add_argument("--max-cer", type=float, default=0.8)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    src_dir = _resolve(args.metadata_dir)
    out_dir = _resolve(args.out_dir)
    if src_dir.resolve() == out_dir.resolve():
        p.error("--out-dir must differ from --metadata-dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, dict] = {}
    for split in args.splits:
        jsonl = src_dir / f"{split}.jsonl"
        rows = load_jsonl(jsonl)
        screen_path = out_dir / f"screen_{split}.jsonl"
        if args.reuse:
            stats = load_jsonl(screen_path)
            if [s["clip_id"] for s in stats] != [r["clip_id"] for r in rows]:
                raise ValueError(f"{screen_path} does not match {jsonl}; rerun without --reuse")
        else:
            log.info("%s: decoding %d clips with %s", split, len(rows), args.ckpt)
            hyps, times = decode_split(rows, jsonl, cfg, _resolve(args.ckpt), args.batch_seconds)
            stats = [clip_stats(r, h, t) for r, h, t in zip(rows, hyps, times)]
            write_jsonl(screen_path, stats)

        flagged, kept = [], []
        by_corpus: dict[str, Counter] = defaultdict(Counter)
        for row, s in zip(rows, stats):
            f = flags_for(s, args)
            corpus = s["corpus"] or "(none)"
            by_corpus[corpus]["clips"] += 1
            by_corpus[corpus]["hours"] += s["duration_s"] / 3600
            if f:
                flagged.append({**s, "flags": f})
                by_corpus[corpus]["flagged"] += 1
                by_corpus[corpus]["flagged_hours"] += s["duration_s"] / 3600
                for name in f:
                    by_corpus[corpus][name] += 1
            else:
                kept.append({**row, "word_times": s["word_times"]} if s.get("word_times") else row)
        flagged.sort(key=lambda s: (-len(s["flags"]), -s["cer"]))
        write_jsonl(out_dir / f"flagged_{split}.jsonl", flagged)
        write_jsonl(out_dir / f"{split}.jsonl", kept)
        n_aligned = sum(1 for s in stats if s.get("word_times"))
        report[split] = {
            "clips": len(rows),
            "flagged": len(flagged),
            "aligned": n_aligned,
            "by_corpus": {
                c: {k: round(v, 2) if isinstance(v, float) else v for k, v in cnt.items()}
                for c, cnt in sorted(by_corpus.items())
            },
        }
        log.info("%s: flagged %d/%d clips, word timings on %d", split, len(flagged), len(rows), n_aligned)
        for c, cnt in sorted(by_corpus.items()):
            log.info(
                "  %-14s flagged %6d/%-7d (%.1f/%.1f h)  fast=%d short_hyp=%d garbage=%d",
                c,
                cnt["flagged"],
                cnt["clips"],
                cnt["flagged_hours"],
                cnt["hours"],
                cnt["fast"],
                cnt["short_hyp"],
                cnt["garbage"],
            )

    for src in sorted(src_dir.glob("*.jsonl")):
        if src.stem not in args.splits:
            shutil.copy2(src, out_dir / src.name)
    save_json(
        out_dir / "screen_report.json",
        {
            "source": str(args.metadata_dir),
            "ckpt": str(args.ckpt),
            "thresholds": {
                "max_cps": args.max_cps,
                "short_hyp_cer": args.short_hyp_cer,
                "min_len_ratio": args.min_len_ratio,
                "max_cer": args.max_cer,
            },
            "splits": report,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
