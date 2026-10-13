#!/usr/bin/env python3
"""No-retrain checks for the three Zipformer failure modes.

Decoder recovery (CTC greedy, RNN-T, char LM, word LM), CTC top-5 on
blanked spans, a chunk-size sweep (16 / 32 / 64 / full context), and the
word LM word-list gap. Chunk -1 is the full-context mask the causal model was
also trained on. Alpha and beta stay at the dev-tuned values.

    python -m evaluation.diagnose_failures
"""

from __future__ import annotations

import argparse
import json
import math

import jiwer
import torch
import torch.nn.functional as F

from decoding.greedy import greedy_decode
from decoding.kenlm import build_decoder, labels_from_tokenizer, load_unigrams
from decoding.unit_lm import decode_batch as unit_lm_decode_batch
from decoding.unit_lm import load_unit_model
from decoding.zipformer_greedy import greedy_search_batch
from eval_youtube_timed import FRAME_S, kenlm_words
from eval_zipformer import _pin_streaming
from eval_zipformer_stream import (
    CHAR_ALPHA,
    CHAR_BEAM,
    CHAR_BETA,
    CHAR_LM_DEFAULT,
    CHAR_TOP_K,
    _find_lm,
    _load_wav,
    _resolve,
    stream_utterance,
)
from evaluation.error_taxonomy import _align
from evaluation.geez_eval import normalize as geez_normalize
from evaluation.inspect_internals import youtube_words
from evaluation.wer import score_texts
from models.zipformer_icefall import build_zipformer, load_icefall
from train import load_cmvn, load_tokenizer
from utils import ROOT, load_jsonl, load_yaml, save_json, setup_logging, write_jsonl

log = setup_logging()

CHUNKS = (16, 32, 64, -1)
CLIPS = (
    (
        "ebc",
        "exp/zipformer_s_streaming_pool/youtube_timed/YiB33Piqh20.wav",
        "exp/zipformer_s_streaming_pool/youtube_timed/yt_subs/YiB33Piqh20.am.srt",
    ),
    (
        "etv",
        "exp/zipformer_s_streaming_pool/youtube_timed/byvT14SJHHI.wav",
        "exp/zipformer_s_streaming_pool/youtube_timed/yt_subs/byvT14SJHHI.am.srt",
    ),
)


def _counts(ref: str, hyp: str) -> dict:
    r = geez_normalize(ref, fold_geez=True)
    h = geez_normalize(hyp, fold_geez=True)
    if not r.strip():
        return {}
    proc = jiwer.process_words(r, h)
    scored = score_texts([r], [h])
    return {
        "wer": round(float(scored["wer"]), 4),
        "cer": round(float(scored["cer"]), 4),
        "substitutions": int(proc.substitutions),
        "deletions": int(proc.deletions),
        "insertions": int(proc.insertions),
        "hits": int(proc.hits),
    }


def _fates(ref_words: list[str], hyp_text: str) -> list[dict]:
    ref = [w for w in ref_words if w]
    hyp = geez_normalize(hyp_text, fold_geez=True).split()
    ops = _align(ref, hyp)
    rows = []
    for kind, rw, hw, extra in ops:
        for w in rw:
            rows.append({"word": w, "kind": kind, "hyp": " ".join(hw), "ed": extra})
    return rows


def _span_top(probs: torch.Tensor, tokenizer, t0: float, t1: float) -> dict:
    n = probs.size(0)
    a = max(0, int(t0 / FRAME_S))
    b = min(n, max(a + 1, int(math.ceil(t1 / FRAME_S))))
    sl = probs[a:b]
    blank_id = int(tokenizer.blank_id)
    blank = sl[:, blank_id]
    other = sl.clone()
    other[:, blank_id] = 0
    best_p, best_i = other.max(dim=-1)
    # Mode of the strongest non-blank letter on this span.
    ids, counts = torch.unique(best_i, return_counts=True)
    mode = int(ids[int(counts.argmax())])
    return {
        "n_frames": int(b - a),
        "blank_p": round(float(blank.mean()), 3),
        "best_nonblank_p": round(float(best_p.mean()), 3),
        "frac_nonblank_gt_0.1": round(float((best_p > 0.1).float().mean()), 3),
        "top_letter": tokenizer.decode([mode], collapse=False),
    }


def _decode(model, tokenizer, encoder_out, n_frames, kenlm, char_pack, *, char: bool):
    # Streaming encoder states are inference tensors. The CTC linear must
    # run under inference mode or autograd refuses to touch them.
    with torch.inference_mode():
        lens = torch.tensor([n_frames], device=encoder_out.device)
        logits = model.ctc_output[1](encoder_out[:, :n_frames])
        rnnt_ids = greedy_search_batch(model, encoder_out[:, :n_frames], lens)[0]
    logits = logits.clone()
    out = {
        "ctc": greedy_decode(logits, tokenizer, lens)[0],
        "rnnt": tokenizer.decode(rnnt_ids, collapse=False),
    }
    text, _words = kenlm_words(kenlm, logits, n_frames, 100, frame_s=FRAME_S)
    out["word_lm"] = text
    if char and char_pack is not None:
        km, umodel = char_pack
        out["char_lm"] = unit_lm_decode_batch(
            logits,
            lens,
            tokenizer,
            km,
            umodel,
            unit="char",
            beam_size=CHAR_BEAM,
            token_top_k=CHAR_TOP_K,
            alpha=CHAR_ALPHA,
            beta=CHAR_BETA,
        )[0]
    return out, logits


def _encode_chunk(model, feats: torch.Tensor, chunk: int, device) -> tuple[torch.Tensor, int]:
    left = -1 if chunk < 0 else 128
    _pin_streaming(model, chunk, left)
    x = feats.unsqueeze(0).to(device)
    lens = torch.tensor([feats.size(0)], dtype=torch.long, device=device)
    with torch.inference_mode():
        enc, enc_lens = model.forward_encoder(x, lens)
    return enc, int(enc_lens[0])


def _oov(fates: list[dict], vocab: set[str]) -> dict:
    subs = [r for r in fates if r["kind"] in ("wrong_word", "near_miss")]
    if not subs:
        return {"n": 0}
    ref_oov = sum(1 for r in subs if r["word"] not in vocab)
    hyp_oov = 0
    for r in subs:
        hyp = r["hyp"].split()
        if hyp and hyp[0] not in vocab:
            hyp_oov += 1
    return {
        "n_sub": len(subs),
        "ref_oov": ref_oov,
        "ref_oov_rate": round(ref_oov / len(subs), 3),
        "hyp_oov": hyp_oov,
        "examples_oov": [r for r in subs if r["word"] not in vocab][:12],
    }


def diagnose_clip(name, wav_path, srt_path, model, tokenizer, cmvn, device, kenlm, char_pack, vocab) -> dict:
    # Chunk sweeps overwrite these. Streaming init reads left_context_frames[0].
    model.encoder.chunk_size = (16, 32, 64, -1)
    model.encoder.left_context_frames = (128, 256, -1)
    wav, sr = _load_wav(_resolve(wav_path))
    ref_words = youtube_words(_resolve(srt_path))
    ref_text = [w["text"] for w in ref_words]
    ref_join = " ".join(ref_text)
    log.info("%s wav %.1fs ref words %d", name, wav.shape[0] / sr, len(ref_words))

    with torch.inference_mode():
        sess = stream_utterance(model, cmvn, wav, sr, device, 0)
    feats = sess.feats.detach().float().cpu()
    enc = sess.encoder_out
    n = int(enc.size(1))
    texts, logits = _decode(model, tokenizer, enc, n, kenlm, char_pack, char=True)
    probs = F.softmax(logits[0].float(), dim=-1).detach().cpu()

    fates = {dec: _fates(ref_text, hyp) for dec, hyp in texts.items()}
    for dec, rows in fates.items():
        if len(rows) != len(ref_text):
            log.warning("%s %s fate rows %d vs ref %d", name, dec, len(rows), len(ref_text))

    word_rows = fates["word_lm"]
    blanked = []
    for i, (row, src) in enumerate(zip(word_rows, ref_words)):
        if row["kind"] not in ("drop", "drop_tail"):
            continue
        stat = _span_top(probs, tokenizer, src["start_s"], src["end_s"])
        if stat["blank_p"] < 0.7:
            continue
        recovery = {
            dec: fates[dec][i]["kind"] if i < len(fates[dec]) else "missing"
            for dec in fates
        }
        blanked.append(
            {
                "i": i,
                "word": row["word"],
                "t": round(src["start_s"], 2),
                **stat,
                "recovery": recovery,
            }
        )

    chunks = {}
    for chunk in CHUNKS:
        label = "full" if chunk < 0 else str(chunk)
        try:
            chunk_enc, chunk_n = _encode_chunk(model, feats, chunk, device)
            chunk_text, _logits = _decode(
                model, tokenizer, chunk_enc, chunk_n, kenlm, char_pack, char=False
            )
        except RuntimeError as exc:
            log.warning("%s chunk %s failed: %s", name, label, exc)
            torch.cuda.empty_cache()
            chunks[label] = {"error": str(exc).split("\n")[0][:240]}
            continue
        del chunk_enc
        torch.cuda.empty_cache()
        entry = {"scores": {dec: _counts(ref_join, hyp) for dec, hyp in chunk_text.items()}}
        if blanked:
            recovered = {}
            for dec, hyp in chunk_text.items():
                rows = _fates(ref_text, hyp)
                n_eq = n_part = 0
                for item in blanked:
                    pos = item["i"]
                    kind = rows[pos]["kind"] if pos < len(rows) else "missing"
                    if kind == "equal":
                        n_eq += 1
                    elif kind == "near_miss":
                        n_part += 1
                recovered[dec] = {"equal": n_eq, "near_miss": n_part, "of": len(blanked)}
            entry["blanked_recovered"] = recovered
        chunks[label] = entry
        sc = entry["scores"].get("ctc", {})
        log.info(
            "%s chunk %s ctc WER %.3f del %s sub %s",
            name,
            label,
            sc.get("wer", -1),
            sc.get("deletions"),
            sc.get("substitutions"),
        )

    dec_scores = {dec: _counts(ref_join, hyp) for dec, hyp in texts.items()}
    kind_counts = {}
    for dec, rows in fates.items():
        kind_counts[dec] = {}
        for row in rows:
            kind_counts[dec][row["kind"]] = kind_counts[dec].get(row["kind"], 0) + 1

    available = [b for b in blanked if b["best_nonblank_p"] >= 0.1 or b["frac_nonblank_gt_0.1"] >= 0.25]
    return {
        "name": name,
        "ref_words": len(ref_words),
        "streaming_scores": dec_scores,
        "kind_counts": kind_counts,
        "n_blanked": len(blanked),
        "n_blanked_letter_available": len(available),
        "blanked": blanked,
        "oov": _oov(word_rows, vocab),
        "chunks": chunks,
    }


def fleurs_chunks(model, tokenizer, cmvn, kenlm, device, n: int) -> dict:
    jsonl = ROOT / "metadata" / "fleurs_am_et" / "test.jsonl"
    if n <= 0 or not jsonl.exists():
        return {"skipped": True}
    from torch.utils.data import DataLoader

    from data.dataset import AsrDataset, collate

    rows = load_jsonl(jsonl)[:n]
    tmp = ROOT / "exp" / "zipformer_s_streaming_pool" / "diagnosis" / "_fleurs_slice.jsonl"
    write_jsonl(tmp, rows)
    ds = AsrDataset(tmp, tokenizer, train=False, cmvn=cmvn, max_duration=None)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate, num_workers=0)
    batches = list(loader)
    out = {"n": len(rows)}
    for chunk in (32, -1):
        label = "full" if chunk < 0 else str(chunk)
        left = -1 if chunk < 0 else 128
        _pin_streaming(model, chunk, left)
        refs, ctc_hyps, word_hyps = [], [], []
        try:
            for batch in batches:
                feats = batch["feats"].to(device)
                lens = batch["feat_lens"].to(device)
                with torch.inference_mode():
                    enc, enc_lens = model.forward_encoder(feats, lens)
                    logits = model.ctc_output[1](enc)
                ctc_hyps.extend(greedy_decode(logits, tokenizer, enc_lens))
                word_hyps.extend(
                    [
                        kenlm_words(kenlm, logits[b : b + 1], int(enc_lens[b]), 100, frame_s=FRAME_S)[0]
                        for b in range(logits.size(0))
                    ]
                )
                refs.extend(batch["text"])
        except RuntimeError as exc:
            log.warning("fleurs chunk %s failed: %s", label, exc)
            torch.cuda.empty_cache()
            out[label] = {"error": str(exc).split("\n")[0][:240]}
            continue
        out[label] = {
            "ctc": _counts(" ".join(geez_normalize(r, fold_geez=True) for r in refs), " ".join(ctc_hyps))
            if False
            else None,
        }
        # Score utterance-by-utterance so a short ref is not glued into one string.
        def _macro(hyps):
            proc_s = proc_d = proc_i = proc_h = 0
            wers = []
            for ref, hyp in zip(refs, hyps):
                c = _counts(ref, hyp)
                if not c:
                    continue
                proc_s += c["substitutions"]
                proc_d += c["deletions"]
                proc_i += c["insertions"]
                proc_h += c["hits"]
                wers.append(c["wer"])
            denom = proc_s + proc_d + proc_i + proc_h
            return {
                "wer_micro": round((proc_s + proc_d + proc_i) / denom, 4) if denom else None,
                "substitutions": proc_s,
                "deletions": proc_d,
                "insertions": proc_i,
                "hits": proc_h,
                "n": len(wers),
            }

        out[label] = {"ctc": _macro(ctc_hyps), "word_lm": _macro(word_hyps)}
        log.info("fleurs n=%d chunk %s ctc %s", n, label, out[label]["ctc"])
        torch.cuda.empty_cache()
    return out


def etv_windows(model, tokenizer, cmvn, device, kenlm) -> list[dict]:
    """Chunk 32 vs full context on short ETV slices. The 291 s clip does not fit this GPU."""
    from models.features import log_mel

    from decoding.zipformer_stream import _bandlimit

    wav, sr = _load_wav(
        _resolve("exp/zipformer_s_streaming_pool/youtube_timed/byvT14SJHHI.wav")
    )
    audio = torch.from_numpy(wav)
    feats = cmvn(log_mel(_bandlimit(audio), dither=0.0)).float().cpu()
    # (start_s, end_s, words the streaming word LM blanked inside the slice)
    spans = (
        (75.0, 100.0, ["እና", "ህግ"]),
        (140.0, 160.0, ["ፓርቲ", "ጉዳይ", "ላይ", "ምንም"]),
        (218.0, 232.0, ["ኮሮኑ"]),
    )
    rows = []
    hop = 100  # 10 ms features
    for a, b, words in spans:
        sl = feats[int(a * hop) : int(b * hop)]
        row = {"t0": a, "t1": b, "words": words, "chunks": {}}
        for chunk in (32, -1):
            label = "full" if chunk < 0 else "32"
            try:
                enc, n = _encode_chunk(model, sl, chunk, device)
                texts, _logits = _decode(model, tokenizer, enc, n, kenlm, None, char=False)
            except RuntimeError as exc:
                torch.cuda.empty_cache()
                row["chunks"][label] = {"error": str(exc).split("\n")[0][:180]}
                continue
            del enc
            torch.cuda.empty_cache()
            present = {}
            for dec, hyp in texts.items():
                norm = geez_normalize(hyp, fold_geez=True)
                present[dec] = {w: w in norm.split() for w in words}
            row["chunks"][label] = present
            log.info("etv window %.0f-%.0f chunk %s %s", a, b, label, present)
        rows.append(row)
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/zipformer_s_streaming_pool.yaml")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--fleurs-n", type=int, default=12)
    p.add_argument("--windows-only", action="store_true")
    p.add_argument("--out", default="exp/zipformer_s_streaming_pool/diagnosis/failures.json")
    args = p.parse_args()

    cfg = load_yaml(_resolve(args.config))
    exp_dir = _resolve(cfg["paths"]["exp_dir"])
    ckpt = _resolve(args.ckpt) if args.ckpt else exp_dir / "best.pt"
    load_icefall()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(cfg, exp_dir)
    cmvn = load_cmvn(cfg, exp_dir, compute_if_missing=False)
    model = build_zipformer(cfg, tokenizer.vocab_size)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    model.to(device).eval()

    lm_path, uni_path = _find_lm(None)
    unigrams = load_unigrams(uni_path) or []
    vocab = {geez_normalize(w, fold_geez=True) for w in unigrams}
    kenlm = build_decoder(
        labels_from_tokenizer(tokenizer),
        str(lm_path),
        unigrams=unigrams,
        alpha=0.5,
        beta=0.8,
    )
    char_km, char_model, char_path = load_unit_model(_resolve(CHAR_LM_DEFAULT))
    log.info("word lm %s (%d) char lm %s", lm_path, len(vocab), char_path)

    out = _resolve(args.out)
    if args.windows_only:
        result = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
        result["etv_windows"] = etv_windows(model, tokenizer, cmvn, device, kenlm)
        save_json(out, result)
        log.info("wrote %s", out)
        return 0

    result = {"ckpt": str(ckpt), "clips": []}
    for name, wav, srt in CLIPS:
        clip = diagnose_clip(
            name, wav, srt, model, tokenizer, cmvn, device, kenlm, (char_km, char_model), vocab
        )
        result["clips"].append(clip)
        save_json(out, result)
        log.info(
            "%s blanked %d letter-available %d oov %s",
            name,
            clip["n_blanked"],
            clip["n_blanked_letter_available"],
            clip["oov"],
        )

    result["fleurs"] = fleurs_chunks(model, tokenizer, cmvn, kenlm, device, args.fleurs_n)
    save_json(out, result)
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
