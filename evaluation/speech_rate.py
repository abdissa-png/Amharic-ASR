"""Speaking rate vs WER, CER, and dropped words.

Two views, both Ge'ez-folded:

1. Observational. Existing Zipformer eval hyps, binned by natural Fidel
   letters per second of speech. Fast speech here is whoever talked quickly,
   so content and rate are entangled.

2. Controlled. The same clips time-stretched. Training only saw 0.9–1.1, so
   1.2+ is out of distribution. Content is fixed; WER movement is from rate.

    python -m evaluation.speech_rate

    python -m evaluation.speech_rate --perturb 0.8,1.0,1.2,1.35 \\
        --jsonl metadata/fleurs_am_et/test.jsonl
"""

from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluation.error_taxonomy import CATS, analyse_pair
from evaluation.geez_eval import normalize as geez_normalize
from evaluation.wer import score_texts
from utils import ROOT, load_jsonl, save_json, setup_logging, write_jsonl

log = setup_logging()

SETS = (
    ("Dataset.ET", "metadata/test.jsonl", "eval_metadata_test_chunk32_{decoder}"),
    ("FLEURS", "metadata/fleurs_am_et/test.jsonl", "eval_fleurs_am_et_test_chunk32_{decoder}"),
    ("BDU", "metadata/bdu_am/test.jsonl", "eval_bdu_am_test_chunk32_{decoder}"),
)
DECODERS = ("ctc_charlm", "ctc_kenlm", "ctc", "ctc_morphlm", "rnnt")
BINS = ("slow", "mid", "fast")
MIN_SPEECH_S = 0.4


def _quantile(xs: list[float], q: float) -> float:
    ys = sorted(xs)
    if not ys:
        return 0.0
    i = min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))
    return ys[i]


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 8:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx < 1e-12 or vy < 1e-12:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    return _pearson(_ranks(xs), _ranks(ys))


def speech_seconds(row: dict) -> float:
    for key in ("speech_s", "duration_s"):
        val = float(row.get(key) or 0.0)
        if val > 0:
            return val
    return 0.0


def rate_of(row: dict) -> dict[str, float] | None:
    sec = speech_seconds(row)
    if sec < MIN_SPEECH_S:
        return None
    ref = geez_normalize(row.get("text_raw") or row["text"], fold_geez=True)
    words = ref.split()
    chars = ref.replace(" ", "")
    if not words or not chars:
        return None
    return {
        "speech_s": sec,
        "n_words": float(len(words)),
        "n_chars": float(len(chars)),
        "wps": len(words) / sec,
        "cps": len(chars) / sec,
    }


def _utt_scores(row: dict, hyp: str) -> dict[str, Any]:
    ref = row.get("text_raw") or row["text"]
    folded = score_texts([ref], [hyp], fold_geez=True)
    tax = analyse_pair(ref, hyp)
    nref = tax["nref"] or 1
    ops = tax["ops"]
    return {
        "wer": folded["wer"],
        "cer": folded["cer"],
        "nref": tax["nref"],
        "errors": tax["errors"],
        **{k: int(ops.get(k, 0)) for k in CATS},
        "drop_rate": ops.get("drop_mid", 0) / nref,
        "drop_tail_rate": ops.get("drop_tail", 0) / nref,
        "near_rate": ops.get("near_miss", 0) / nref,
        "wrong_rate": ops.get("wrong_word", 0) / nref,
    }


def _summarize(records: list[dict]) -> dict[str, Any]:
    if not records:
        return {"n_utt": 0}
    nref = sum(r["nref"] for r in records)
    errors = sum(r["errors"] for r in records)
    ops = {k: sum(r[k] for r in records) for k in CATS}
    return {
        "n_utt": len(records),
        "n_ref_words": nref,
        "wer_pct": round(100 * errors / nref, 2) if nref else 0.0,
        "cer_pct": round(100 * sum(r["cer"] * r["n_chars"] for r in records) / max(sum(r["n_chars"] for r in records), 1), 2),
        "mean_cps": round(sum(r["cps"] for r in records) / len(records), 2),
        "mean_wps": round(sum(r["wps"] for r in records) / len(records), 2),
        "mean_speech_s": round(sum(r["speech_s"] for r in records) / len(records), 2),
        "mean_n_words": round(sum(r["n_words"] for r in records) / len(records), 1),
        "drop_mid_per_100w": round(100 * ops["drop_mid"] / nref, 2) if nref else 0.0,
        "drop_tail_per_100w": round(100 * ops["drop_tail"] / nref, 2) if nref else 0.0,
        "near_miss_per_100w": round(100 * ops["near_miss"] / nref, 2) if nref else 0.0,
        "wrong_word_per_100w": round(100 * ops["wrong_word"] / nref, 2) if nref else 0.0,
        "share_of_errors_pct": {k: round(100 * ops[k] / errors, 1) if errors else 0.0 for k in CATS},
    }


def analyse_hyps(name: str, rows: list[dict], hyps: dict[str, str], decoder: str) -> dict[str, Any]:
    recs: list[dict] = []
    for row in rows:
        hyp = hyps.get(row["clip_id"])
        if hyp is None:
            continue
        rates = rate_of(row)
        if rates is None:
            continue
        rec = {"clip_id": row["clip_id"], **rates, **_utt_scores(row, hyp)}
        recs.append(rec)
    cps = [r["cps"] for r in recs]
    lo, hi = _quantile(cps, 1 / 3), _quantile(cps, 2 / 3)
    med = _quantile(cps, 0.5)
    for r in recs:
        r["bin"] = "slow" if r["cps"] <= lo else ("fast" if r["cps"] > hi else "mid")
        r["faster_than_median"] = r["cps"] > med
    bins = {b: _summarize([r for r in recs if r["bin"] == b]) for b in BINS}
    vs_median = {
        "slower_or_equal": _summarize([r for r in recs if not r["faster_than_median"]]),
        "faster": _summarize([r for r in recs if r["faster_than_median"]]),
    }
    corr = {
        "pearson_cps_wer": _pearson(cps, [r["wer"] for r in recs]),
        "pearson_cps_cer": _pearson(cps, [r["cer"] for r in recs]),
        "pearson_cps_drop_rate": _pearson(cps, [r["drop_rate"] for r in recs]),
        "spearman_cps_wer": _spearman(cps, [r["wer"] for r in recs]),
        "spearman_cps_cer": _spearman(cps, [r["cer"] for r in recs]),
        "spearman_cps_drop_rate": _spearman(cps, [r["drop_rate"] for r in recs]),
    }
    for k, v in list(corr.items()):
        corr[k] = None if v is None else round(v, 3)
    out = {
        "name": name,
        "decoder": decoder,
        "n_utt": len(recs),
        "median_cps": round(med, 2),
        "tercile_cps": [round(lo, 2), round(hi, 2)],
        "overall": _summarize(recs),
        "bins": bins,
        "vs_median": vs_median,
        "corr": corr,
    }
    _log_block(out)
    return out


def _log_block(rep: dict) -> None:
    log.info(
        "%s  %s  n=%d  median %.2f chars/s  terciles %s",
        rep["name"],
        rep["decoder"],
        rep["n_utt"],
        rep["median_cps"],
        rep["tercile_cps"],
    )
    for b in BINS:
        s = rep["bins"][b]
        if not s.get("n_utt"):
            continue
        log.info(
            "  %-5s n=%3d  %.2f c/s  WER %.2f CER %.2f  drop_mid/100w %.2f  near %.2f  wrong %.2f",
            b,
            s["n_utt"],
            s["mean_cps"],
            s["wer_pct"],
            s["cer_pct"],
            s["drop_mid_per_100w"],
            s["near_miss_per_100w"],
            s["wrong_word_per_100w"],
        )
    c = rep["corr"]
    log.info(
        "  corr cps↔WER p=%s s=%s   cps↔CER p=%s s=%s   cps↔drop p=%s s=%s",
        c["pearson_cps_wer"],
        c["spearman_cps_wer"],
        c["pearson_cps_cer"],
        c["spearman_cps_cer"],
        c["pearson_cps_drop_rate"],
        c["spearman_cps_drop_rate"],
    )


def observational(exp: Path, decoders: list[str]) -> list[dict]:
    reports = []
    for name, jsonl, folder_t in SETS:
        meta = ROOT / jsonl
        if not meta.exists():
            continue
        rows = load_jsonl(meta)
        for decoder in decoders:
            folder = exp / folder_t.format(decoder=decoder)
            hyp_path = folder / "hypotheses.jsonl"
            if not hyp_path.exists():
                continue
            hyps = {r["clip_id"]: r.get("text", "") for r in load_jsonl(hyp_path)}
            reports.append(analyse_hyps(name, rows, hyps, decoder))
    return reports


def _reuse_natural(exp: Path, jsonl: Path, rows: list[dict]) -> dict[str, list[str]] | None:
    """Skip a second 1.0× decode when the usual eval folders already exist."""
    tag = f"eval_{jsonl.parent.name}_{jsonl.stem}_chunk32_{{decoder}}"
    out: dict[str, list[str]] = {}
    for decoder in ("ctc", "ctc_kenlm", "ctc_charlm", "ctc_morphlm"):
        path = exp / tag.format(decoder=decoder) / "hypotheses.jsonl"
        if not path.exists():
            continue
        hyps = {r["clip_id"]: r.get("text", "") for r in load_jsonl(path)}
        if all(r["clip_id"] in hyps for r in rows):
            out[decoder] = [hyps[r["clip_id"]] for r in rows]
            log.info("  reuse %s", path)
    return out or None


def _decode_speed(model, tokenizer, cmvn, args, jsonl: Path, device, speed: float) -> dict[str, list[str]]:
    """One encoder pass at this time-stretch. Greedy CTC plus any loaded LMs."""
    import torch
    from torch.utils.data import DataLoader

    from data.dataset import AsrDataset, collate
    from decoding.greedy import greedy_decode
    from decoding.kenlm import decode_batch as kenlm_decode_batch
    from decoding.unit_lm import decode_batch as unit_lm_decode_batch
    from eval_zipformer import _pin_streaming

    ds = AsrDataset(jsonl, tokenizer, train=False, cmvn=cmvn, max_duration=None, eval_speed=speed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=2)
    _pin_streaming(model, args.chunk_size, args.left_context)
    out: dict[str, list[str]] = {"ctc": []}
    if args.kenlm_decoder is not None:
        out["ctc_kenlm"] = []
    for spec in args.unit_lms:
        out[spec["name"]] = []
    clip_ids: list[str] = []
    t0 = time.perf_counter()
    n_done = 0
    for batch in loader:
        feats = batch["feats"].to(device, non_blocking=True)
        lens = batch["feat_lens"].to(device, non_blocking=True)
        with torch.inference_mode():
            encoder_out, encoder_out_lens = model.forward_encoder(feats, lens)
            logits = model.ctc_output[1](encoder_out)
        out["ctc"].extend(greedy_decode(logits, tokenizer, encoder_out_lens))
        if args.kenlm_decoder is not None:
            out["ctc_kenlm"].extend(
                kenlm_decode_batch(args.kenlm_decoder, logits, encoder_out_lens, beam_width=args.beam_width)
            )
        for spec in args.unit_lms:
            for i in range(logits.size(0)):
                t = int(encoder_out_lens[i])
                out[spec["name"]].append(
                    unit_lm_decode_batch(
                        logits[i : i + 1, :t].float().cpu(),
                        torch.tensor([t]),
                        tokenizer,
                        spec["km"],
                        spec["model"],
                        unit=spec["unit"],
                        beam_size=args.unit_beam,
                        token_top_k=args.unit_top_k,
                        alpha=spec["alpha"],
                        beta=spec["beta"],
                    )[0]
                )
        clip_ids.extend(batch["clip_id"])
        n_done += len(batch["clip_id"])
        if n_done % 64 == 0 or n_done == len(ds):
            rate = n_done / max(time.perf_counter() - t0, 1e-6)
            log.info("  speed=%.2f  %d/%d  %.1f utt/s", speed, n_done, len(ds), rate)
    order = {cid: i for i, cid in enumerate(clip_ids)}
    rows = load_jsonl(jsonl)
    return {name: [hyps[order[r["clip_id"]]] for r in rows] for name, hyps in out.items()}


def perturb(args, jsonl: Path, exp: Path) -> list[dict]:
    import torch

    from decoding.kenlm import build_decoder, labels_from_tokenizer, load_unigrams
    from decoding.unit_lm import load_unit_model
    from eval_zipformer import _find_lm, _resolve
    from models.zipformer_icefall import build_zipformer, load_icefall
    from train import load_cmvn, load_tokenizer
    from utils import load_yaml

    cfg = load_yaml(_resolve(args.config))
    ckpt_path = _resolve(args.ckpt)
    load_icefall()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(cfg, exp)
    cmvn = load_cmvn(cfg, exp, compute_if_missing=False)
    model = build_zipformer(cfg, tokenizer.vocab_size)
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    model.to(device).eval()
    args.chunk_size = 32
    args.left_context = 128
    args.batch_size = args.batch_size
    args.kenlm_decoder = None
    args.unit_lms = []
    if not args.no_kenlm:
        lm_path, uni_path = _find_lm(args.lm)
        args.kenlm_decoder = build_decoder(
            labels_from_tokenizer(tokenizer),
            str(lm_path),
            unigrams=load_unigrams(uni_path),
            alpha=0.5,
            beta=0.8,
        )
        log.info("perturb word LM %s", lm_path)
    if args.char_lm:
        km, umodel, path = load_unit_model(_resolve(args.char_lm))
        args.unit_lms.append(
            {
                "name": "ctc_charlm",
                "km": km,
                "model": umodel,
                "path": path,
                "unit": "char",
                "alpha": 0.4,
                "beta": 1.2,
            }
        )
        log.info("perturb char LM %s", path)

    rows = load_jsonl(jsonl)
    corpus = jsonl.parent.name
    factors = args.perturb
    by_speed: dict[str, dict[str, Any]] = {}
    for factor in factors:
        if factor > 1:
            log.info("time-stretch ×%.2f  (%.0f%% faster)", factor, 100 * (factor - 1))
        elif factor < 1:
            log.info("time-stretch ×%.2f  (%.0f%% slower)", factor, 100 * (1 - factor))
        else:
            log.info("time-stretch ×%.2f  (natural)", factor)
        if abs(factor - 1.0) < 1e-6:
            hyps_by = _reuse_natural(exp, jsonl, rows)
            if hyps_by is None:
                hyps_by = _decode_speed(model, tokenizer, cmvn, args, jsonl, device, factor)
        else:
            hyps_by = _decode_speed(model, tokenizer, cmvn, args, jsonl, device, factor)
        speed_rep: dict[str, Any] = {}
        for decoder, hyps in hyps_by.items():
            hyp_map = {r["clip_id"]: h for r, h in zip(rows, hyps)}
            tag = f"{corpus}_{jsonl.stem}_speed{factor:g}_{decoder}"
            out_dir = exp / "speech_rate" / f"eval_{tag}"
            write_jsonl(out_dir / "hypotheses.jsonl", [{"clip_id": r["clip_id"], "text": h} for r, h in zip(rows, hyps)])
            # For stretched audio the *effective* rate is natural_cps * factor.
            fake_rows = []
            for r in rows:
                rr = dict(r)
                sec = speech_seconds(r)
                if sec > 0:
                    rr["speech_s"] = sec / factor
                    rr["duration_s"] = float(r.get("duration_s") or sec) / factor
                fake_rows.append(rr)
            speed_rep[decoder] = analyse_hyps(f"{corpus}×{factor:g}", fake_rows, hyp_map, decoder)
            save_json(out_dir / "metrics.json", speed_rep[decoder])
        by_speed[f"{factor:g}"] = speed_rep
    return [{"kind": "perturb", "jsonl": str(jsonl), "factors": factors, "by_speed": by_speed}]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp-dir", default="exp/zipformer_s_streaming_pool")
    p.add_argument("--decoder", default=",".join(DECODERS), help="comma-separated; missing folders are skipped")
    p.add_argument("--perturb", default=None, help="comma-separated speed factors, e.g. 0.8,1.0,1.2,1.35")
    p.add_argument("--jsonl", default="metadata/fleurs_am_et/test.jsonl")
    p.add_argument("--config", default="configs/zipformer_s_streaming_pool.yaml")
    p.add_argument("--ckpt", default="exp/zipformer_s_streaming_pool/best.pt")
    p.add_argument("--char-lm", default="exp/zipformer_s_streaming_pool/lm_char")
    p.add_argument("--lm", default=None)
    p.add_argument("--no-kenlm", action="store_true")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--beam-width", type=int, default=100)
    p.add_argument("--unit-beam", type=int, default=32)
    p.add_argument("--unit-top-k", type=int, default=12)
    args = p.parse_args()
    exp = ROOT / args.exp_dir if not Path(args.exp_dir).is_absolute() else Path(args.exp_dir)
    decoders = [d.strip() for d in args.decoder.split(",") if d.strip()]
    reports = observational(exp, decoders)
    perturb_reports: list[dict] = []
    if args.perturb:
        args.perturb = [float(x) for x in args.perturb.split(",") if x.strip()]
        jsonl = Path(args.jsonl)
        if not jsonl.is_absolute():
            jsonl = ROOT / jsonl
        perturb_reports = perturb(args, jsonl, exp)
    out = exp / "speech_rate.json"
    save_json(out, {"observational": reports, "perturb": perturb_reports})
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
