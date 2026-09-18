"""Full experiment metrics: WER/CER/MER plus runtime and model size."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from evaluation.geez_eval import normalize as geez_normalize
from evaluation.unseen_words import seen_unseen_wer
from evaluation.wer import score_texts
from data.oov import utterance_buckets, vocab_from_rows


def peak_rss_mb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        try:
            import resource

            # Linux reports ru_maxrss in kilobytes.
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            return float("nan")


def file_size_mb(path: str | Path | None) -> float | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return round(p.stat().st_size / (1024 * 1024), 2)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def score_split(
    rows: list[dict[str, Any]],
    hyps: Sequence[str],
    *,
    v_train: set[str] | None = None,
    fold_geez: bool = True,
) -> dict[str, Any]:
    refs = [r.get("text_raw") or r["text"] for r in rows]
    overall = score_texts(refs, hyps, fold_geez=fold_geez)
    out: dict[str, Any] = {"overall": overall}
    if v_train is None:
        return out
    out["seen_unseen"] = seen_unseen_wer(refs, hyps, v_train, fold_geez=fold_geez)
    buckets = utterance_buckets(rows, v_train)
    id_to_idx = {r["clip_id"]: i for i, r in enumerate(rows)}
    out["subsets"] = {}
    for name, clip_ids in buckets.items():
        if name == "ALL":
            continue
        idxs = [id_to_idx[c] for c in clip_ids if c in id_to_idx]
        if not idxs:
            continue
        out["subsets"][name] = score_texts(
            [refs[i] for i in idxs],
            [hyps[i] for i in idxs],
            fold_geez=fold_geez,
        )
        out["subsets"][name]["n_utt"] = len(idxs)
    return out


def timed_transcribe(
    utterances: Iterable[tuple[Any, float]],
    decode_fn: Callable[[Any], str],
) -> tuple[list[str], dict[str, float]]:
    """``utterances`` yields (payload, duration_seconds)."""
    hyps: list[str] = []
    latencies: list[float] = []
    audio_s = 0.0
    t0 = time.perf_counter()
    rss0 = peak_rss_mb()
    peak = rss0
    for payload, dur in utterances:
        audio_s += dur
        s = time.perf_counter()
        hyps.append(decode_fn(payload))
        latencies.append(time.perf_counter() - s)
        peak = max(peak, peak_rss_mb())
    wall = time.perf_counter() - t0
    latencies.sort()
    n = max(1, len(latencies))
    runtime = {
        "n": len(hyps),
        "audio_seconds": round(audio_s, 3),
        "wall_seconds": round(wall, 3),
        "rtf": round(wall / audio_s, 4) if audio_s else float("nan"),
        "latency_mean_s": round(sum(latencies) / n, 4),
        "latency_p50_s": round(latencies[n // 2], 4),
        "latency_p95_s": round(latencies[min(n - 1, int(0.95 * n))], 4),
        "ram_mb_start": round(rss0, 1),
        "ram_mb_peak": round(peak, 1),
    }
    return hyps, runtime
