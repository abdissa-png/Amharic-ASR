#!/usr/bin/env python3
"""Break a Zipformer eval's WER into split, drop, near-miss, and wrong-word.

    python -m evaluation.error_taxonomy \\
        --exp-dir exp/zipformer_s_streaming_pool \\
        --decoder ctc_kenlm
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from evaluation.geez_eval import normalize as geez_normalize
from utils import ROOT, load_jsonl, save_json, setup_logging

log = setup_logging()

SETS = (
    ("Dataset.ET", "metadata/test.jsonl", "eval_metadata_test_chunk32_{decoder}"),
    ("FLEURS", "metadata/fleurs_am_et/test.jsonl", "eval_fleurs_am_et_test_chunk32_{decoder}"),
    ("BDU", "metadata/bdu_am/test.jsonl", "eval_bdu_am_test_chunk32_{decoder}"),
)

# Categories consume this many WER operations (S+D+I).
CATS = (
    "near_miss",
    "split",
    "fusion",
    "wrong_word",
    "drop_mid",
    "drop_tail",
    "insert",
)


def _ed(a: str, b: str) -> int:
    if a == b:
        return 0
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * m
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[m]


def _same_syllable(a: str, b: str) -> bool:
    """True if two Ge'ez letters share a consonant and differ only in vowel order."""
    if len(a) != 1 or len(b) != 1:
        return False
    oa, ob = ord(a), ord(b)
    if not (0x1200 <= oa <= 0x137C and 0x1200 <= ob <= 0x137C):
        return False
    return oa // 8 == ob // 8


def _char_pairs(ref: str, hyp: str) -> list[tuple[str, str]]:
    n, m = len(ref), len(hyp)
    prev = list(range(m + 1))
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    for j in range(1, m + 1):
        bt[0][j] = "I"
    row = prev
    for i, ca in enumerate(ref, 1):
        cur = [i] + [0] * m
        bt[i][0] = "D"
        for j, cb in enumerate(hyp, 1):
            costs = (
                (row[j] + 1, "D"),
                (cur[j - 1] + 1, "I"),
                (row[j - 1] + (ca != cb), "S" if ca != cb else "M"),
            )
            cur[j], bt[i][j] = min(costs, key=lambda x: x[0])
        row = cur
    i, j = n, m
    pairs: list[tuple[str, str]] = []
    while i > 0 or j > 0:
        op = bt[i][j]
        if op in ("S", "M"):
            if op == "S":
                pairs.append((ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif op == "D":
            pairs.append((ref[i - 1], ""))
            i -= 1
        else:
            pairs.append(("", hyp[j - 1]))
            j -= 1
    pairs.reverse()
    return pairs


def _align(ref_w: list[str], hyp_w: list[str]) -> list[tuple]:
    """Word alignment that prefers split/fusion concatenations over raw edits."""
    n, m = len(ref_w), len(hyp_w)
    inf = n + m + 1
    cost = [[inf] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0
    for i in range(n + 1):
        for j in range(m + 1):
            if i < n and cost[i][j] + 1 < cost[i + 1][j]:
                cost[i + 1][j] = cost[i][j] + 1
                back[i + 1][j] = ("drop", i, j, 1, 0)
            if j < m and cost[i][j] + 1 < cost[i][j + 1]:
                cost[i][j + 1] = cost[i][j] + 1
                back[i][j + 1] = ("insert", i, j, 0, 1)
            if i < n and j < m:
                r, h = ref_w[i], hyp_w[j]
                if r == h:
                    c = cost[i][j]
                    if c < cost[i + 1][j + 1]:
                        cost[i + 1][j + 1] = c
                        back[i + 1][j + 1] = ("equal", i, j, 1, 1)
                else:
                    ed = _ed(r, h)
                    kind = "near_miss" if 0 < ed <= 2 else "wrong_word"
                    c = cost[i][j] + 1
                    if c < cost[i + 1][j + 1]:
                        cost[i + 1][j + 1] = c
                        back[i + 1][j + 1] = (kind, i, j, 1, 1, ed)
            if i < n:
                acc = ""
                for k in range(2, min(6, m - j) + 1):
                    acc += hyp_w[j + k - 1]
                    if acc == ref_w[i]:
                        c = cost[i][j] + k
                        if c < cost[i + 1][j + k]:
                            cost[i + 1][j + k] = c
                            back[i + 1][j + k] = ("split", i, j, 1, k)
                        break
                    if len(acc) > len(ref_w[i]):
                        break
            if j < m:
                acc = ""
                for k in range(2, min(6, n - i) + 1):
                    acc += ref_w[i + k - 1]
                    if acc == hyp_w[j]:
                        c = cost[i][j] + k
                        if c < cost[i + k][j + 1]:
                            cost[i + k][j + 1] = c
                            back[i + k][j + 1] = ("fusion", i, j, k, 1)
                        break
                    if len(acc) > len(hyp_w[j]):
                        break
    ops = []
    i, j = n, m
    while i or j:
        rec = back[i][j]
        kind, si, sj, dr, dh, *rest = rec
        ops.append((kind, ref_w[si : si + dr], hyp_w[sj : sj + dh], rest[0] if rest else None))
        i, j = si, sj
    ops.reverse()
    if ops and all(k == "drop" for k, *_ in ops[-1:]):
        pass
    tail = 0
    k = len(ops)
    while k > 0 and ops[k - 1][0] == "drop":
        tail += len(ops[k - 1][1])
        k -= 1
    if tail >= 2 or (tail and n and tail / n >= 0.2):
        kept, dropped = ops[:k], ops[k:]
        refs = [w for _, rw, *_ in dropped for w in rw]
        ops = kept + [("drop_tail", refs, [], None)]
    return ops


def _wer_ops(kind: str, nref: int, nhyp: int) -> int:
    if kind == "equal":
        return 0
    if kind == "drop":
        return nref
    if kind == "drop_tail":
        return nref
    if kind == "insert":
        return nhyp
    if kind == "split":
        return nhyp  # 1 sub + (k-1) ins
    if kind == "fusion":
        return nref  # 1 sub + (k-1) del
    return 1  # near_miss / wrong_word


def analyse_pair(ref: str, hyp: str) -> dict:
    r = geez_normalize(ref, fold_geez=True)
    h = geez_normalize(hyp, fold_geez=True)
    rw, hw = r.split(), h.split()
    ops = _align(rw, hw)
    counts = Counter()
    events = Counter()
    pairs = Counter()
    letters = Counter()
    dropped = Counter()
    inserted = Counter()
    nref = len(rw)
    for kind, rws, hws, extra in ops:
        n_ops = _wer_ops(kind, len(rws), len(hws))
        bucket = kind if kind in CATS else ("drop_mid" if kind == "drop" else kind)
        if bucket == "drop":
            bucket = "drop_mid"
        counts[bucket] += n_ops
        events[bucket] += 1
        if kind in ("near_miss", "wrong_word") and rws and hws:
            pairs[(kind, rws[0], hws[0])] += 1
            if kind == "near_miss":
                for a, b in _char_pairs(rws[0], hws[0]):
                    if a != b:
                        letters[(a, b)] += 1
        if kind in ("drop", "drop_tail"):
            for w in rws:
                dropped[w] += 1
        if kind == "insert":
            for w in hws:
                inserted[w] += 1
        if kind == "split":
            pairs[("split", rws[0], " ".join(hws))] += 1
        if kind == "fusion":
            pairs[("fusion", " ".join(rws), hws[0])] += 1
    hyp_prefix = bool(rw and hw and nref >= 4 and rw[: len(hw)] == hw and len(hw) < nref)
    return {
        "nref": nref,
        "nhyp": len(hw),
        "ops": dict(counts),
        "events": dict(events),
        "pairs": pairs,
        "letters": letters,
        "dropped": dropped,
        "inserted": inserted,
        "hyp_prefix": hyp_prefix,
        "empty_hyp": not hw,
        "errors": sum(counts.values()),
    }


def _top(counter: Counter, n: int = 12) -> list[dict]:
    out = []
    for key, c in counter.most_common(n):
        if isinstance(key, tuple) and len(key) == 3:
            out.append({"kind": key[0], "ref": key[1], "hyp": key[2], "n": c})
        elif isinstance(key, tuple) and len(key) == 2:
            out.append({"ref": key[0], "hyp": key[1], "n": c})
        else:
            out.append({"word": key, "n": c})
    return out


def analyse_set(name: str, rows: list[dict], hyps: dict[str, str]) -> dict:
    tot = Counter()
    events = Counter()
    pairs: Counter = Counter()
    letters: Counter = Counter()
    dropped: Counter = Counter()
    inserted: Counter = Counter()
    nref = n_utt = prefix = empty = errors = 0
    examples = defaultdict(list)
    for row in rows:
        cid = row["clip_id"]
        ref = row.get("text_raw") or row["text"]
        hyp = hyps.get(cid, "")
        a = analyse_pair(ref, hyp)
        n_utt += 1
        nref += a["nref"]
        errors += a["errors"]
        tot.update(a["ops"])
        events.update(a["events"])
        pairs.update(a["pairs"])
        letters.update(a["letters"])
        dropped.update(a["dropped"])
        inserted.update(a["inserted"])
        prefix += int(a["hyp_prefix"])
        empty += int(a["empty_hyp"])
        for kind, count in a["ops"].items():
            if count and len(examples[kind]) < 4:
                examples[kind].append(
                    {
                        "clip_id": cid,
                        "ref": geez_normalize(ref)[:180],
                        "hyp": geez_normalize(hyp)[:180],
                    }
                )
    wer = errors / nref if nref else 0.0
    share = {k: round(100 * tot[k] / errors, 1) if errors else 0.0 for k in CATS}
    return {
        "name": name,
        "n_utt": n_utt,
        "n_ref_words": nref,
        "n_errors": errors,
        "wer_pct": round(100 * wer, 2),
        "share_of_errors_pct": share,
        "events": dict(events),
        "utt_hyp_is_prefix_pct": round(100 * prefix / n_utt, 1) if n_utt else 0.0,
        "empty_hyp_pct": round(100 * empty / n_utt, 1) if n_utt else 0.0,
        "top_confusions": _top(pairs, 15),
        "top_letter_confusions": _top(letters, 15),
        "top_dropped": _top(dropped, 12),
        "top_inserted": _top(inserted, 12),
        "examples": {k: v for k, v in examples.items()},
        "counts": dict(tot),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", default="exp/zipformer_s_streaming_pool")
    p.add_argument("--decoder", default="ctc_kenlm")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    exp = ROOT / args.exp_dir
    reports = []
    for name, jsonl, folder_t in SETS:
        folder = exp / folder_t.format(decoder=args.decoder)
        hyp_path = folder / "hypotheses.jsonl"
        meta = ROOT / jsonl
        if not hyp_path.exists() or not meta.exists():
            log.warning("skip %s (%s or %s missing)", name, hyp_path, meta)
            continue
        hyps = {r["clip_id"]: r.get("text", "") for r in load_jsonl(hyp_path)}
        rows = load_jsonl(meta)
        log.info("scoring %s  %d utt  %s", name, len(rows), args.decoder)
        reports.append(analyse_set(name, rows, hyps))
        s = reports[-1]
        log.info(
            "  WER %.2f  near=%.1f split=%.1f fusion=%.1f wrong=%.1f mid-drop=%.1f tail-drop=%.1f ins=%.1f",
            s["wer_pct"],
            s["share_of_errors_pct"]["near_miss"],
            s["share_of_errors_pct"]["split"],
            s["share_of_errors_pct"]["fusion"],
            s["share_of_errors_pct"]["wrong_word"],
            s["share_of_errors_pct"]["drop_mid"],
            s["share_of_errors_pct"]["drop_tail"],
            s["share_of_errors_pct"]["insert"],
        )
    out = Path(args.out) if args.out else exp / f"error_taxonomy_{args.decoder}.json"
    if not out.is_absolute():
        out = ROOT / out
    save_json(out, {"decoder": args.decoder, "sets": reports})
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
