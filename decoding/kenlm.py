"""CTC + n-gram LM decoding via pyctcdecode (Decoder C).

Train the LM first::

    python -m decoding.train_lm --out-dir exp/20m_char_4layer/lm

Then::

    python evaluate.py --config configs/20m_char_4layer.yaml \\
        --ckpt exp/20m_char_4layer/best.pt --split dev --decoder kenlm
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from utils import ROOT, setup_logging

log = setup_logging()


def labels_from_tokenizer(tokenizer) -> list[str]:
    """pyctcdecode labels: blank is \"\", every other id is the emitted string."""
    if not hasattr(tokenizer, "tokens"):
        raise TypeError("tokenizer must expose .tokens for KenLM decoding")
    labels = list(tokenizer.tokens)
    labels[int(tokenizer.blank_id)] = ""
    is_bpe = any(s.startswith("\u2581") or s.startswith("##") for s in labels)
    if not is_bpe:
        unk = getattr(tokenizer, "unk_id", None)
        if unk is not None:
            labels[int(unk)] = "\ufffd"
        pad = tokenizer.stoi.get("<pad>") if hasattr(tokenizer, "stoi") else None
        if pad is not None:
            labels[int(pad)] = "\u200b"
    return labels


def load_unigrams(path: Path | None) -> list[str] | None:
    if path is None or not Path(path).exists():
        return None
    words = [w.strip() for w in Path(path).read_text(encoding="utf-8").splitlines() if w.strip()]
    return words or None


def lm_files(lm_dir: Path) -> tuple[Path | None, Path | None]:
    model = None
    for name in ("lm.bin", "lm.arpa"):
        cand = Path(lm_dir) / name
        if cand.exists():
            model = cand
            break
    uni = Path(lm_dir) / "unigrams.txt"
    return model, uni if uni.exists() else None


def resolve_lm_dir(exp_dir: Path) -> tuple[Path | None, Path | None]:
    """Prefer ``lm_word/`` (word 5-gram) over the smaller speech-only ``lm/``."""
    for folder in ("lm_word", "lm"):
        model, uni = lm_files(Path(exp_dir) / folder)
        if model is not None:
            return model, uni
    return None, None


def _ensure_kenlm_bindings():
    """Load pip ``kenlm.Model``, not a local C++ clone named ``kenlm/``.

    Cloning https://github.com/kpu/kenlm into the repo root makes
    ``import kenlm`` succeed with a module that has no ``Model``.
    """
    import kenlm as km

    if hasattr(km, "Model"):
        return km
    shadow = None
    if getattr(km, "__path__", None):
        shadow = km.__path__[0]
    elif getattr(km, "__file__", None):
        shadow = km.__file__
    log.warning("import kenlm resolved to %s (no Model); retrying site-packages", shadow)
    for key in list(sys.modules):
        if key == "kenlm" or key.startswith("kenlm."):
            del sys.modules[key]
    saved = sys.path[:]
    root = ROOT.resolve()
    sys.path = [p for p in sys.path if Path(p).resolve() != root]
    try:
        km = importlib.import_module("kenlm")
    except ImportError as exc:
        raise ImportError(
            "pip install kenlm  (Python bindings; separate from compiling lmplz). "
            "Move the C++ clone off the repo root: mkdir -p third_party && mv kenlm third_party/kenlm"
        ) from exc
    finally:
        sys.path[:] = saved
    sys.modules["kenlm"] = km
    if not hasattr(km, "Model"):
        raise ImportError(
            "kenlm.Model is missing. pip install kenlm, and move the C++ clone "
            "off the repo root: mkdir -p third_party && mv kenlm third_party/kenlm"
        )
    log.info("using kenlm bindings from %s", getattr(km, "__file__", km))
    return km


def build_decoder(
    labels: Sequence[str],
    lm_path: str | None,
    *,
    unigrams: Sequence[str] | None = None,
    alpha: float = 0.5,
    beta: float = 1.5,
):
    try:
        from pyctcdecode import build_ctcdecoder
    except ImportError as exc:
        raise ImportError("pip install pyctcdecode kenlm") from exc
    if lm_path:
        try:
            _ensure_kenlm_bindings()
        except ImportError as exc:
            raise ImportError(
                "pyctcdecode needs the kenlm Python bindings: pip install kenlm"
            ) from exc
    return build_ctcdecoder(
        list(labels),
        kenlm_model_path=str(lm_path) if lm_path else None,
        unigrams=list(unigrams) if unigrams else None,
        alpha=alpha,
        beta=beta,
    )


def decode_batch(
    decoder,
    logits,
    lengths,
    *,
    beam_width: int = 100,
) -> list[str]:
    """``logits``: (B, T, V) raw acoustic logits (pyctcdecode log-softmaxes internally)."""
    arr = logits.detach().float().cpu().numpy() if hasattr(logits, "detach") else np.asarray(logits)
    hyps: list[str] = []
    for b in range(arr.shape[0]):
        t = int(lengths[b])
        hyps.append(decoder.decode(arr[b, :t], beam_width=beam_width))
    return hyps
