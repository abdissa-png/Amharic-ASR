"""E1 — pretrained Amharic / multilingual baselines on Dataset.ET test.

    python -m baselines.run --model ethio-asr --split test
    python -m baselines.run --model all --split test

Models:
  dvoice       speechbrain/asr-wav2vec2-dvoice-amharic
  ethio-asr    badrex/Ethio-ASR-amharic
  omnilingual  omniASR_CTC_300M_v2
  whisper      openai/whisper-small
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

import numpy as np

from data.oov import vocab_from_rows
from data.prepare import load_split_jsonl
from evaluation.metrics import file_size_mb, peak_rss_mb, score_split
from utils import ROOT, save_json, setup_logging, write_jsonl

log = setup_logging()

MODELS = ("dvoice", "ethio-asr", "omnilingual", "whisper")


def _load_audio(row: dict, root: Path = ROOT) -> np.ndarray:
    import soundfile as sf

    path = root / row["audio"]
    if path.exists():
        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            raise ValueError(f"{path} is {sr} Hz")
        return audio
    raise FileNotFoundError(path)


def transcribe_ethio(rows: list[dict], batch_size: int = 8) -> list[str]:
    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    mid = "badrex/Ethio-ASR-amharic"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForCTC.from_pretrained(mid).to(device).eval()
    hyps: list[str] = []
    with torch.inference_mode():
        for i in range(0, len(rows), batch_size):
            chunk = [_load_audio(r) for r in rows[i : i + batch_size]]
            inp = proc(chunk, sampling_rate=16000, return_tensors="pt", padding=True)
            key = "input_features" if "input_features" in inp else "input_values"
            kwargs = {key: inp[key].to(device)}
            if "attention_mask" in inp:
                kwargs["attention_mask"] = inp["attention_mask"].to(device)
            logits = model(**kwargs).logits
            hyps.extend(proc.batch_decode(logits.argmax(-1).cpu().numpy()))
            log.info("ethio-asr %d/%d", min(i + batch_size, len(rows)), len(rows))
    return [t.strip() for t in hyps]


def transcribe_whisper(rows: list[dict], batch_size: int = 8, model_id: str = "openai/whisper-small") -> list[str]:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device).eval()
    hyps: list[str] = []
    with torch.inference_mode():
        for i in range(0, len(rows), batch_size):
            chunk = [_load_audio(r) for r in rows[i : i + batch_size]]
            feats = proc(chunk, sampling_rate=16000, return_tensors="pt").input_features.to(device)
            ids = model.generate(feats, language="am", task="transcribe", max_new_tokens=440, num_beams=1)
            hyps.extend(proc.batch_decode(ids, skip_special_tokens=True))
            log.info("whisper %d/%d", min(i + batch_size, len(rows)), len(rows))
    return [t.strip() for t in hyps]


def transcribe_dvoice(rows: list[dict]) -> list[str]:
    from speechbrain.inference.ASR import EncoderASR

    asr = EncoderASR.from_hparams(
        source="speechbrain/asr-wav2vec2-dvoice-amharic",
        savedir=str(ROOT / "exp" / "pretrained" / "dvoice-amharic"),
        run_opts={"device": "cuda"} if _cuda() else None,
    )
    hyps = []
    for i, row in enumerate(rows):
        path = str(ROOT / row["audio"])
        hyps.append(asr.transcribe_file(path).strip())
        if i % 50 == 0:
            log.info("dvoice %d/%d", i, len(rows))
    return hyps


def transcribe_omnilingual(rows: list[dict], batch_size: int = 4) -> list[str]:
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    pipeline = ASRInferencePipeline(model_card="omniASR_CTC_300M_v2")
    hyps: list[str] = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        audio = [
            {"waveform": _load_audio(r), "sample_rate": 16000} for r in chunk
        ]
        out = pipeline.transcribe(audio, batch_size=len(audio))
        hyps.extend([str(t).strip() for t in out])
        log.info("omnilingual %d/%d", min(i + batch_size, len(rows)), len(rows))
    return hyps


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


RUNNERS: dict[str, Callable] = {
    "ethio-asr": transcribe_ethio,
    "whisper": transcribe_whisper,
    "dvoice": transcribe_dvoice,
    "omnilingual": transcribe_omnilingual,
}


def run(model_name: str, split: str, out_dir: Path) -> dict:
    rows = load_split_jsonl(split)
    t0 = time.perf_counter()
    rss0 = peak_rss_mb()
    hyps = RUNNERS[model_name](rows)
    wall = time.perf_counter() - t0
    audio_s = sum(r["duration_s"] for r in rows)
    v_train = set(vocab_from_rows(load_split_jsonl("train")))
    scores = score_split(rows, hyps, v_train=v_train)
    result = {
        "model": model_name,
        "split": split,
        "scores": scores,
        "runtime": {
            "wall_seconds": wall,
            "audio_seconds": audio_s,
            "rtf": wall / audio_s if audio_s else float("nan"),
            "ram_mb_start": rss0,
            "ram_mb_peak": peak_rss_mb(),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "metrics.json", result)
    write_jsonl(
        out_dir / "hypotheses.jsonl",
        [{"clip_id": r["clip_id"], "text": h} for r, h in zip(rows, hyps)],
    )
    log.info(
        "%s %s WER=%.2f CER=%.2f RTF=%.3f",
        model_name,
        split,
        scores["overall"]["wer_pct"],
        scores["overall"]["cer_pct"],
        result["runtime"]["rtf"],
    )
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ethio-asr", choices=["all", *MODELS])
    p.add_argument("--split", default="test")
    p.add_argument("--out-dir", type=Path, default=ROOT / "exp" / "baselines")
    args = p.parse_args()
    names = MODELS if args.model == "all" else (args.model,)
    for name in names:
        try:
            run(name, args.split, args.out_dir / name / args.split)
        except Exception as exc:  # noqa: BLE001
            log.error("%s failed: %s: %s", name, type(exc).__name__, exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
