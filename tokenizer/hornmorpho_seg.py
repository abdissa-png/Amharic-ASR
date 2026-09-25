"""Mine Amharic prefixes/stems/suffixes with HornMorpho, then surface-align them.

HornMorpho returns *canonical* morphemes (ኦች, ኣል) that often do not concatenate
to the written word. CTC labels must. We locate the stem inside the surface
form (allowing Fidel vowel-order allomorphy) and take the leftover prefix and
suffix *as written*.

HornMorpho is a vocab miner only. Inference still uses longest-match on the
surface piece inventory (no `hm` at decode time).

    python -m tokenizer.hornmorpho_seg
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from phonology.decomposition import _FIDEL_BASES
from utils import ROOT, setup_logging

log = setup_logging()

_HOMOPHONE_FIRST = {
    "ሀ": "ሐኀ",
    "ሐ": "ሀኀ",
    "ኀ": "ሀሐ",
    "ሰ": "ሠ",
    "ሠ": "ሰ",
    "አ": "ኣዐ",
    "ኣ": "አዐ",
    "ዐ": "አኣ",
    "ጸ": "ፀ",
    "ፀ": "ጸ",
}


def _order_alts(word: str) -> list[str]:
    if not word:
        return []
    cp = ord(word[-1])
    for base in _FIDEL_BASES:
        if base <= cp < base + 8:
            return [word[:-1] + chr(base + o) for o in range(8) if base + o != cp]
    return []


def _stem_candidates(stem: str, lemma: str | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def add(s: str) -> None:
        if s and s not in seen:
            seen.add(s)
            out.append(s)

    for raw in (stem, (stem or "").replace("-", ""), lemma or ""):
        add(raw)
        for alt in _order_alts(raw):
            add(alt)
        if raw:
            first, rest = raw[0], raw[1:]
            for sub in _HOMOPHONE_FIRST.get(first, ""):
                add(sub + rest)
                for alt in _order_alts(sub + rest):
                    add(alt)
    return out


def surface_align(word: str, stem: str | None, lemma: str | None) -> dict | None:
    """Return surface prefix/stem/suffix that concatenate to *word*, or None."""
    if not word:
        return None
    for cand in _stem_candidates(stem or "", lemma):
        i = word.find(cand)
        if i < 0:
            continue
        pre, suf = word[:i], word[i + len(cand) :]
        if pre + cand + suf != word:
            continue
        return {"prefix": pre, "stem": cand, "suffix": suf, "matched": cand}
    return None


def _serialize(word: str, analysis) -> dict:
    rec: dict = {"word": word, "pos": None, "lemma": None, "seg": None, "stem": None}
    if not analysis:
        return rec
    a = analysis[0]
    rec["pos"] = a.get("pos")
    rec["lemma"] = a.get("lemma")
    rec["seg"] = a.get("seg")
    stem = a.get("stem")
    if isinstance(stem, dict):
        rec["stem"] = stem.get("seg")
    elif isinstance(stem, str):
        rec["stem"] = stem
    return rec


def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[rec["word"]] = rec
    return out


def append_cache(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def analyze_words(words: list[str], cache_path: Path) -> dict[str, dict]:
    import hm

    cache = load_cache(cache_path)
    missing = [w for w in words if w not in cache]
    log.info("HornMorpho cache %d/%d words already analyzed", len(words) - len(missing), len(words))
    t0 = time.time()
    for i, w in enumerate(missing, 1):
        try:
            ana = hm.anal("a", w)
        except Exception as exc:  # noqa: BLE001 — analyzer can fail on junk tokens
            log.warning("hm.anal failed on %r: %s", w, exc)
            ana = None
        rec = _serialize(w, ana)
        cache[w] = rec
        append_cache(cache_path, rec)
        if i == 1 or i % 25 == 0 or i == len(missing):
            log.info("analyzed %d/%d  last=%s  elapsed=%.0fs", i, len(missing), w, time.time() - t0)
    try:
        hm.exit(save=True)
    except Exception:
        pass
    return cache


def mine(cache: dict[str, dict], word_counts: Counter[str], *, min_affix: int, max_affix_len: int) -> dict:
    prefixes: Counter[str] = Counter()
    suffixes: Counter[str] = Counter()
    stems: Counter[str] = Counter()
    aligned = 0
    analyzed = 0
    examples = []
    unaligned_examples = []
    for word, rec in cache.items():
        count = word_counts[word]
        if rec.get("seg") or rec.get("stem") or rec.get("lemma"):
            analyzed += 1
        hit = surface_align(word, rec.get("stem"), rec.get("lemma"))
        if not hit:
            if len(unaligned_examples) < 20 and rec.get("seg"):
                unaligned_examples.append({"word": word, "seg": rec.get("seg"), "stem": rec.get("stem")})
            continue
        aligned += 1
        if hit["prefix"]:
            prefixes[hit["prefix"]] += count
        if hit["suffix"]:
            suffixes[hit["suffix"]] += count
        stems[hit["stem"]] += count
        if len(examples) < 25:
            examples.append({"word": word, **hit, "seg": rec.get("seg"), "count": count})

    def keep(counter: Counter[str], *, prefix: bool) -> list[dict]:
        rows = []
        for aff, n in counter.most_common():
            if n < min_affix or not (1 <= len(aff) <= max_affix_len):
                continue
            rows.append({"surface": aff, "count": n, "kind": "prefix" if prefix else "suffix"})
        return rows

    return {
        "n_cached": len(cache),
        "n_with_analysis": analyzed,
        "n_surface_aligned": aligned,
        "align_rate": round(aligned / max(1, len(cache)), 3),
        "prefixes": keep(prefixes, prefix=True),
        "suffixes": keep(suffixes, prefix=False),
        "n_stems": len(stems),
        "top_stems": stems.most_common(40),
        "align_examples": examples,
        "unaligned_examples": unaligned_examples,
        "prefix_counter": prefixes,
        "suffix_counter": suffixes,
        "stem_counter": stems,
    }


def write_stem_corpus(
    path: Path,
    cache: dict[str, dict],
    word_counts: Counter[str],
    *,
    cap_per_type: int = 20,
) -> dict:
    """Write a BPE corpus: aligned stems, else the original word (verbs we couldn't split)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_stem = n_full = 0
    with path.open("w", encoding="utf-8") as fh:
        for word, rec in cache.items():
            count = min(word_counts[word], cap_per_type)
            if count <= 0:
                continue
            hit = surface_align(word, rec.get("stem"), rec.get("lemma"))
            token = hit["stem"] if hit else word
            if hit:
                n_stem += count
            else:
                n_full += count
            fh.write((token + "\n") * count)
    return {"path": str(path), "stem_lines": n_stem, "fullword_lines": n_full}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--speech-text", type=Path, default=ROOT / "exp" / "20m_bpe" / "corpus_1024.txt")
    p.add_argument("--top-n", type=int, default=2500)
    p.add_argument("--min-affix", type=int, default=15)
    p.add_argument("--max-affix-len", type=int, default=5)
    p.add_argument("--out-dir", type=Path, default=ROOT / "exp" / "tokenizers" / "hm_bpe")
    args = p.parse_args()

    counts = Counter(args.speech_text.read_text(encoding="utf-8").split())
    top = [w for w, _ in counts.most_common(args.top_n)]
    cache_path = args.out_dir / "hm_cache.jsonl"
    cache = analyze_words(top, cache_path)
    report = mine(cache, counts, min_affix=args.min_affix, max_affix_len=args.max_affix_len)
    stem_info = write_stem_corpus(args.out_dir / "stem_corpus.txt", cache, counts)
    dump = {k: v for k, v in report.items() if k not in {"prefix_counter", "suffix_counter", "stem_counter"}}
    dump["stem_corpus"] = stem_info
    dump["top_n"] = args.top_n
    out = args.out_dir / "hm_mine.json"
    out.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("wrote %s", out)
    print("align_rate", dump["align_rate"], "aligned", dump["n_surface_aligned"], "/", dump["n_cached"])
    print("prefixes", " ".join(r["surface"] for r in dump["prefixes"][:30]))
    print("suffixes", " ".join(r["surface"] for r in dump["suffixes"][:30]))
    print("top stems", " ".join(s for s, _ in dump["top_stems"][:20]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
