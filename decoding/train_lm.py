"""Train an n-gram LM from speech transcripts and/or word corpus.

``--unit word`` (default) is the pyctcdecode 5-gram: it scores *words*
(characters concatenated until a space).

``--unit char`` writes each Ge'ez letter as a KenLM token and ``▁`` for
space. Do not feed that ARPA to pyctcdecode; use the prefix beam in
``decoding.unit_lm``.

``--unit morph`` peels surface prefixes/suffixes with
``phonology.decomposition.surface_morphs`` and trains a morph n-gram.
That LM is also scored in ``decoding.unit_lm``, at word boundaries.

Speech-only (decontaminated train transcripts)::

    python -m decoding.train_lm --out-dir exp/20m_char_12layer/lm

External text (e.g. word corpus) mixed with those transcripts::

    python -m decoding.train_lm --text word_corpus.txt \\
        --out-dir exp/20m_char_12layer/lm_word --order 5

Retokenize an existing word ``train.txt`` into a character 6-gram::

    python -m decoding.train_lm --unit char --lm-only --order 6 \\
        --src-text exp/20m_char_12layer/lm_word/train.txt \\
        --out-dir exp/zipformer_s_streaming_pool/lm_char \\
        --prune "0 0 0 1 1 2"

Then evaluate on **dev** first (do not retune α/β on test).
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from data.normalize import for_training
from data.prepare import load_split_jsonl
from phonology.decomposition import surface_morphs
from utils import ROOT, save_json, setup_logging

# KenLM uses space as the token delimiter, so the CTC space character is this.
SPACE_SYM = "\u2581"

log = setup_logging()

BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"
BACKOFF = 0.4  # stupid backoff

# If word corpus has no newlines, split on Amharic / Latin sentence ends.
# ፡፡ is the classic Ethiopic full stop (two wordspaces).
_SENT_SPLIT = re.compile(r"(?:፡፡|።|፧|[?!])+")
_ETHIOPIC = re.compile(r"[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F]")


def heldout_strings() -> set[str]:
    """Exact raw strings plus training-normalised forms from dev/test."""
    held: set[str] = set()
    try:
        splits = [("dev", load_split_jsonl("dev")), ("test", load_split_jsonl("test"))]
    except FileNotFoundError as exc:
        log.warning("no Dataset.ET metadata (%s); skipping held-out decontamination", exc)
        return held
    for _split, rows in splits:
        for row in rows:
            for key in ("text", "text_raw", "text_eval"):
                val = (row.get(key) or "").strip()
                if val:
                    held.add(val)
                    norm = for_training(val)
                    if norm:
                        held.add(norm)
            pid = row.get("prompt_id") or ""
            if pid:
                held.add(pid)
    return held


def decontaminated_texts() -> list[str]:
    """Train transcripts with any exact string that appears in dev/test removed."""
    try:
        train_rows = load_split_jsonl("train")
    except FileNotFoundError as exc:
        log.warning("no train jsonl (%s); not mixing speech transcripts", exc)
        return []
    held = heldout_strings()
    texts: list[str] = []
    skipped = 0
    for row in train_rows:
        if (row.get("prompt_id") or "") in held:
            skipped += 1
            continue
        text = (row.get("text") or "").strip()
        if not text or text in held or for_training(text) in held:
            skipped += 1
            continue
        texts.append(for_training(text) or text)
    log.info("LM speech corpus: %d train lines (%d dropped as overlapping held-out text)", len(texts), skipped)
    return texts


def usable_lm_line(text: str, *, min_words: int = 3, min_ethiopic_frac: float = 0.5) -> bool:
    """Keep Amharic-majority lines; drop boilerplate / Latin dumps / tiny fragments."""
    words = text.split()
    if len(words) < min_words:
        return False
    eth = sum(1 for w in words if _ETHIOPIC.search(w))
    return eth >= 2 and eth / len(words) >= min_ethiopic_frac


def wrap_long(text: str, *, max_words: int = 80, min_words: int = 3) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text] if words else []
    out: list[str] = []
    for i in range(0, len(words), max_words):
        chunk = words[i : i + max_words]
        if len(chunk) >= min_words:
            out.append(" ".join(chunk))
    return out


def split_sentences(blob: str) -> list[str]:
    """Split a punctuation-concatenated blob; last fragment is incomplete if no closer."""
    if not blob:
        return []
    return [p.strip() for p in _SENT_SPLIT.split(blob) if p.strip()]


def iter_raw_sentences(path: Path, *, chunk_chars: int = 2_000_000):
    """Stream a possibly newline-less file and yield sentence-sized fragments."""
    leftover = ""
    with path.open("r", encoding="utf-8") as fh:
        while True:
            chunk = fh.read(chunk_chars)
            if not chunk:
                break
            data = leftover + chunk
            parts = _SENT_SPLIT.split(data)
            leftover = parts[-1]
            for part in parts[:-1]:
                raw = part.strip()
                if raw:
                    yield raw
    tail = leftover.strip()
    if tail:
        yield tail


def clean_sentence(raw: str) -> str:
    """Match CTC / KenLM tokenisation: NFC, punct→space, keep Fidel homophones."""
    return for_training(raw)


def retokenize_line(line: str, unit: str) -> str:
    """Map a word-level training line into KenLM tokens for ``unit``."""
    line = line.strip()
    if not line:
        return ""
    if unit == "word":
        return line
    if unit == "char":
        return " ".join(SPACE_SYM if ch == " " else ch for ch in line)
    if unit == "morph":
        toks: list[str] = []
        for i, word in enumerate(line.split()):
            if i:
                toks.append(SPACE_SYM)
            toks.extend(surface_morphs(word) or [word])
        return " ".join(toks)
    raise ValueError(f"unknown LM unit {unit!r}")


def convert_word_corpus(src: Path, dest: Path, unit: str) -> Counter[str]:
    """Rewrite ``src`` (word lines) into ``dest`` as ``unit`` tokens."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    n_in = n_out = 0
    with src.open("r", encoding="utf-8") as inf, dest.open("w", encoding="utf-8") as outf:
        for line in inf:
            n_in += 1
            toked = retokenize_line(line, unit)
            if not toked:
                continue
            outf.write(toked + "\n")
            counts.update(toked.split())
            n_out += 1
    log.info("retokenized %s -> %s  unit=%s  lines %d -> %d  types=%d", src, dest, unit, n_in, n_out, len(counts))
    return counts


def prune_for_order(prune: str | None, order: int) -> str | None:
    if not prune:
        return None
    parts = prune.split()
    if not parts:
        return None
    if len(parts) > order:
        parts = parts[:order]
    while len(parts) < order:
        parts.append(parts[-1])
    return " ".join(parts)


def iter_cleaned_external(
    path: Path,
    held: set[str],
    *,
    min_words: int = 3,
    min_ethiopic_frac: float = 0.5,
    max_words: int = 80,
):
    n_raw = n_empty = n_held = n_filter = n_keep = 0
    for raw in iter_raw_sentences(path):
        n_raw += 1
        cleaned = clean_sentence(raw)
        if not cleaned:
            n_empty += 1
            continue
        for line in wrap_long(cleaned, max_words=max_words, min_words=min_words):
            if line in held:
                n_held += 1
                continue
            if not usable_lm_line(line, min_words=min_words, min_ethiopic_frac=min_ethiopic_frac):
                n_filter += 1
                continue
            n_keep += 1
            yield line
    log.info(
        "external %s: raw_sents=%d kept=%d empty=%d heldout=%d filtered=%d",
        path.name,
        n_raw,
        n_keep,
        n_empty,
        n_held,
        n_filter,
    )


def _find_bin(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for extra in (
        ROOT / "third_party" / "kenlm" / "build" / "bin" / name,
        ROOT / "kenlm" / "build" / "bin" / name,
        Path.home() / "kenlm" / "build" / "bin" / name,
        Path("/usr/local/bin") / name,
        Path("/tmp/kenlm-src/build/bin") / name,
        Path("/content/kenlm/build/bin") / name,
        Path("/content/kenlm-src/build/bin") / name,
    ):
        if extra.exists():
            return str(extra)
    return None


def write_arpa_stupid_backoff(sentences: list[list[str]], path: Path, order: int = 3) -> None:
    """Tiny ARPA n-gram so decoding works when ``lmplz`` is not installed.

    Prefer KenLM's Kneser–Ney ``lmplz`` when available; this is a fallback.
    """
    order = max(1, min(int(order), 3))
    counts: list[Counter[tuple[str, ...]]] = [Counter() for _ in range(order + 1)]
    for words in sentences:
        padded = [BOS] + words + [EOS]
        for n in range(1, order + 1):
            for i in range(len(padded) - n + 1):
                counts[n][tuple(padded[i : i + n])] += 1
    vocab = {w for (w,) in counts[1]} | {UNK}
    n1 = sum(counts[1].values())
    log10 = math.log10

    def p_log10(gram: tuple[str, ...]) -> float:
        n = len(gram)
        c = counts[n][gram]
        if n == 1:
            return log10((c + 0.1) / (n1 + 0.1 * len(vocab)))
        ctx = gram[:-1]
        cc = counts[n - 1][ctx]
        if c > 0 and cc > 0:
            return log10(c / cc)
        return log10(BACKOFF) + p_log10(gram[1:])

    lines = ["\\data\\"]
    ngram_sizes = []
    grams_by_n: list[list[tuple[str, ...]]] = [[]]
    for n in range(1, order + 1):
        keys = sorted(counts[n].keys())
        if n == 1:
            extra = [(UNK,)]
            keys = sorted(set(keys) | set(extra))
        grams_by_n.append(keys)
        ngram_sizes.append(len(keys))
        lines.append(f"ngram {n}={len(keys)}")
    lines.append("")

    for n in range(1, order + 1):
        lines.append(f"\\{n}-grams:")
        for gram in grams_by_n[n]:
            lp = p_log10(gram)
            token_str = " ".join(gram)
            if n < order:
                lines.append(f"{lp:.6f}\t{token_str}\t{log10(BACKOFF):.6f}")
            else:
                lines.append(f"{lp:.6f}\t{token_str}")
        lines.append("")
    lines.append("\\end\\")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote fallback ARPA %s (order=%d, 1-grams=%d)", path, order, ngram_sizes[0])


def train_kenlm(
    text_path: Path,
    arpa_path: Path,
    bin_path: Path,
    *,
    order: int,
    memory_mb: int = 8192,
    prune: str | None = "0 0 1 1 2",
) -> Path:
    lmplz = _find_bin("lmplz")
    build_binary = _find_bin("build_binary")
    if lmplz is None:
        if text_path.stat().st_size > 20_000_000:
            raise SystemExit(
                "Cleaning finished (train.txt / unigrams.txt) but lmplz was not found.\n"
                "pip install kenlm is only Python bindings. On Colab, skip libboost-all-dev "
                "(it pulls a broken python3.12-dev). Run:\n"
                "  apt-get update\n"
                "  apt-get install -y libboost-program-options-dev libboost-system-dev "
                "libboost-thread-dev libboost-test-dev zlib1g-dev\n"
                "  rm -rf /tmp/kenlm-src\n"
                "  git clone --depth 1 https://github.com/kpu/kenlm.git /tmp/kenlm-src\n"
                "  /usr/bin/cmake -S /tmp/kenlm-src -B /tmp/kenlm-src/build "
                "-DKENLM_MAX_ORDER=6 -DBoost_NO_BOOST_CMAKE=ON\n"
                "  /usr/bin/cmake --build /tmp/kenlm-src/build -j2\n"
                "  export PATH=/tmp/kenlm-src/build/bin:$PATH\n"
                "  which lmplz\n"
                "Then:\n"
                "  python -m decoding.train_lm --out-dir <same-out-dir> --lm-only --order 5 --memory-mb 4096"
            )
        log.warning(
            "lmplz not on PATH. Install KenLM (cmake && make) for a proper 5-gram, "
            "or using the Python 3-gram fallback. See decoding/train_lm.py docstring."
        )
        sentences = [line.split() for line in text_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        write_arpa_stupid_backoff(sentences, arpa_path, order=min(order, 3))
        return arpa_path

    cmd = [
        lmplz,
        "-o",
        str(order),
        "--text",
        str(text_path),
        "--arpa",
        str(arpa_path),
        "--discount_fallback",
        "--memory",
        f"{int(memory_mb)}M",
    ]
    if prune:
        cmd.extend(["--prune", *prune.split()])
    log.info("running %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    if build_binary is None:
        log.warning("build_binary not found; evaluate will load the ARPA (slower)")
        return arpa_path
    subprocess.run([build_binary, str(arpa_path), str(bin_path)], check=True)
    log.info("wrote %s", bin_path)
    return bin_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=ROOT / "exp" / "20m_char_4layer" / "lm")
    p.add_argument("--text", type=Path, default=None, help="Extra corpus (word corpus.txt, etc.). Streamed; newlines optional.")
    p.add_argument("--mix-train", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--order", type=int, default=5, help="n-gram order for lmplz (fallback caps at 3)")
    p.add_argument("--memory-mb", type=int, default=8192)
    p.add_argument("--prune", default="0 0 1 1 2", help="lmplz --prune counts; empty string disables")
    p.add_argument("--max-unigrams", type=int, default=300_000)
    p.add_argument("--unit", choices=("word", "char", "morph"), default="word")
    p.add_argument(
        "--src-text",
        type=Path,
        default=None,
        help="Existing word-level train.txt to retokenize when --unit is char or morph",
    )
    p.add_argument(
        "--lm-only",
        action="store_true",
        help="Skip corpus cleaning; run lmplz on <out-dir>/train.txt (after optional --src-text convert)",
    )
    args = p.parse_args()
    if args.unit != "word" and not args.src_text and not args.lm_only:
        raise SystemExit(
            "--unit char/morph needs --src-text of a word train.txt, "
            "or --lm-only on an already-converted <out-dir>/train.txt"
        )
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    text_path = out / "train.txt"

    if args.src_text:
        src = args.src_text if args.src_text.is_absolute() else ROOT / args.src_text
        if not src.exists():
            raise FileNotFoundError(src)
        if args.unit == "word":
            raise SystemExit("--src-text is for --unit char or morph; word LMs already use train.txt")
        word_counts = convert_word_corpus(src, text_path, args.unit)
        uni = [w for w, _ in word_counts.most_common(args.max_unigrams or None)]
        (out / "unigrams.txt").write_text("\n".join(uni) + "\n", encoding="utf-8")
        log.info("wrote %s (%d types)", out / "unigrams.txt", len(uni))
    elif args.lm_only:
        if not text_path.exists() or text_path.stat().st_size == 0:
            raise FileNotFoundError(f"--lm-only needs an existing corpus at {text_path}")
        log.info("lm-only: using existing %s (%.1f MB)", text_path, text_path.stat().st_size / 1e6)
    else:
        held = heldout_strings()
        texts: list[str] = []
        if args.mix_train:
            texts.extend(decontaminated_texts())

        word_counts: Counter[str] = Counter()
        n_lines = 0
        with text_path.open("w", encoding="utf-8") as fh:
            for line in texts:
                fh.write(line + "\n")
                word_counts.update(line.split())
                n_lines += 1
            if args.text:
                src = args.text if args.text.is_absolute() else ROOT / args.text
                if not src.exists():
                    raise FileNotFoundError(src)
                seen_speech = set(texts)
                seen_hash: set[int] = {hash(s) for s in seen_speech}
                for line in iter_cleaned_external(src, held):
                    h = hash(line)
                    if line in seen_speech or h in seen_hash:
                        continue
                    seen_hash.add(h)
                    fh.write(line + "\n")
                    word_counts.update(line.split())
                    n_lines += 1

        ranked = [w for w, _ in word_counts.most_common()]
        head = ranked[: args.max_unigrams] if args.max_unigrams else ranked
        if args.mix_train and texts:
            have = set(head)
            extra = [w for line in texts for w in line.split() if w not in have]
            seen_w: set[str] = set()
            must: list[str] = []
            for w in extra:
                if w not in seen_w:
                    seen_w.add(w)
                    must.append(w)
            uni = head + must
        else:
            uni = head
        uni_path = out / "unigrams.txt"
        uni_path.write_text("\n".join(uni) + "\n", encoding="utf-8")
        log.info(
            "wrote %s (%d lines) and %s (%d of %d word types)",
            text_path,
            n_lines,
            uni_path,
            len(uni),
            len(word_counts),
        )

    prune = prune_for_order((args.prune or "").strip() or None, args.order)
    model_path = train_kenlm(
        text_path,
        out / "lm.arpa",
        out / "lm.bin",
        order=args.order,
        memory_mb=args.memory_mb,
        prune=prune,
    )
    save_json(
        out / "meta.json",
        {
            "unit": args.unit,
            "order": args.order,
            "space": SPACE_SYM,
            "prune": prune,
            "src_text": str(args.src_text) if args.src_text else None,
            "model": str(model_path),
        },
    )
    log.info("LM ready: %s", model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
