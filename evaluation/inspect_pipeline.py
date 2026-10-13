#!/usr/bin/env python3
"""Spectrogram, front-end conv, and Zipformer attention/conv at drop times.

Streaming Zipformer only. Summaries are per frame (no full attention maps):
attention entropy, mass on the last 160 ms, mass on the left context, peak lag,
and the conv-module residual RMS next to the layer input RMS.

    python -m evaluation.inspect_pipeline \\
        --wav exp/zipformer_s_streaming_pool/youtube_timed/YiB33Piqh20.wav \\
        --ref-srt exp/zipformer_s_streaming_pool/youtube_timed/yt_subs/YiB33Piqh20.am.srt
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from decoding.greedy import greedy_decode
from decoding.kenlm import build_decoder, labels_from_tokenizer, load_unigrams
from decoding.zipformer_greedy import greedy_search_batch
from eval_youtube_timed import FRAME_S, kenlm_words
from eval_zipformer_stream import _find_lm, _load_wav, _resolve, stream_utterance
from evaluation.inspect_internals import CHUNK_FRAMES, youtube_words
from evaluation.error_taxonomy import _align
from evaluation.geez_eval import normalize as geez_normalize
from models.zipformer_icefall import build_zipformer, load_icefall
from train import load_cmvn, load_tokenizer
from utils import load_yaml, save_json, setup_logging

log = setup_logging()

# One streaming step emits 32 embed frames. Conv2d subsample is 2× on 10 ms
# features, so an embed frame is 20 ms and a step is 640 ms.
EMBED_FRAME_S = 0.02
STEP_S = 32 * EMBED_FRAME_S
LOCAL_S = 0.16


def _layers(module):
    inner = getattr(module, "encoder", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner.layers
    return module.layers


def _attach(model) -> dict:
    """Wrap each encoder layer and the conv subsample. Returns live buffers."""
    load_icefall()
    buf = {
        "chunk": 0,
        "embed_rms": [],
        "mel_rms": [],
        "stacks": {},
    }
    ds_of = [int(d) for d in model.encoder.downsampling_factor]

    def embed_wrapped(orig):
        def streaming_forward(x, x_lens, cached_left_pad):
            # x: (N, 77, 80) log-mel. The 32 new embed frames line up with the
            # first 64 feature frames (the shift); the tail is right context.
            mel = x[0, :64].detach().float().pow(2).mean(-1)
            mel_rms = mel.reshape(32, 2).mean(-1)
            out, out_lens, cache = orig(x, x_lens, cached_left_pad)
            rms = out[0].detach().float().pow(2).mean(-1).sqrt()
            buf["mel_rms"].append(mel_rms.cpu().numpy())
            buf["embed_rms"].append(rms.cpu().numpy())
            buf["chunk"] += 1
            return out, out_lens, cache

        return streaming_forward

    model.encoder_embed.streaming_forward = embed_wrapped(model.encoder_embed.streaming_forward)

    def layer_wrapped(layer, stack, li, ds):
        orig = layer.streaming_forward

        def streaming_forward(
            src,
            pos_emb,
            cached_key,
            cached_nonlin_attn,
            cached_val1,
            cached_val2,
            cached_conv1,
            cached_conv2,
            left_context_len,
            src_key_padding_mask,
        ):
            captured = {}
            attn_orig = layer.self_attn_weights.streaming_forward
            c1_orig = layer.conv_module1.streaming_forward
            c2_orig = layer.conv_module2.streaming_forward

            def attn_w(*args, **kwargs):
                weights, cached = attn_orig(*args, **kwargs)
                captured["attn"] = weights
                return weights, cached

            def c1_w(*args, **kwargs):
                y, cache = c1_orig(*args, **kwargs)
                captured["c1"] = y
                return y, cache

            def c2_w(*args, **kwargs):
                y, cache = c2_orig(*args, **kwargs)
                captured["c2"] = y
                return y, cache

            layer.self_attn_weights.streaming_forward = attn_w
            layer.conv_module1.streaming_forward = c1_w
            layer.conv_module2.streaming_forward = c2_w
            try:
                result = orig(
                    src,
                    pos_emb,
                    cached_key,
                    cached_nonlin_attn,
                    cached_val1,
                    cached_val2,
                    cached_conv1,
                    cached_conv2,
                    left_context_len,
                    src_key_padding_mask,
                )
            finally:
                layer.self_attn_weights.streaming_forward = attn_orig
                layer.conv_module1.streaming_forward = c1_orig
                layer.conv_module2.streaming_forward = c2_orig

            # chunk was incremented in the embed wrapper before this stack runs.
            chunk = buf["chunk"] - 1
            frame_s = EMBED_FRAME_S * ds
            attn = captured["attn"].detach().float()[ :, 0].mean(0)  # (T, S)
            t_len, s_len = attn.shape
            left = s_len - t_len
            n_local = max(1, int(round(LOCAL_S / frame_s)))
            ent = -(attn.clamp_min(1e-9) * attn.clamp_min(1e-9).log()).sum(-1)
            local = torch.empty(t_len, device=attn.device)
            far = torch.empty(t_len, device=attn.device)
            lag = torch.empty(t_len, device=attn.device)
            # lag bins in seconds: 0-80 ms, 80-320 ms, 0.32-1.28 s, older
            edges = (0.08, 0.32, 1.28)
            hist = torch.zeros(t_len, 4, device=attn.device)
            for t in range(t_len):
                cur = left + t
                lo = max(0, cur - n_local + 1)
                local[t] = attn[t, lo : cur + 1].sum()
                far[t] = attn[t, :left].sum() if left else 0.0
                peak = int(attn[t].argmax())
                lag[t] = (cur - peak) * frame_s
                lags = (cur - torch.arange(s_len, device=attn.device)) * frame_s
                hist[t, 0] = attn[t, lags < edges[0]].sum()
                hist[t, 1] = attn[t, (lags >= edges[0]) & (lags < edges[1])].sum()
                hist[t, 2] = attn[t, (lags >= edges[1]) & (lags < edges[2])].sum()
                hist[t, 3] = attn[t, lags >= edges[2]].sum()
            src_rms = src.detach().float().pow(2).mean(-1).sqrt()[:, 0]
            conv = captured["c1"].detach().float() + captured["c2"].detach().float()
            conv_rms = conv.pow(2).mean(-1).sqrt()[:, 0]
            times = chunk * STEP_S + (np.arange(t_len) + 0.5) * frame_s
            key = (stack, li)
            slot = buf["stacks"].setdefault(
                key,
                {k: [] for k in ("t", "ent", "local", "far", "lag", "src", "conv", "hist")},
            )
            slot["t"].append(times)
            slot["ent"].append(ent.cpu().numpy())
            slot["local"].append(local.cpu().numpy())
            slot["far"].append(far.cpu().numpy())
            slot["lag"].append(lag.cpu().numpy())
            slot["src"].append(src_rms.cpu().numpy())
            slot["conv"].append(conv_rms.cpu().numpy())
            slot["hist"].append(hist.cpu().numpy())
            return result

        return streaming_forward

    for stack, module in enumerate(model.encoder.encoders):
        for li, layer in enumerate(_layers(module)):
            layer.streaming_forward = layer_wrapped(layer, stack, li, ds_of[stack])
    buf["ds"] = ds_of
    return buf


def _cat(buf) -> dict:
    embed_rms = np.concatenate(buf["embed_rms"]) if buf["embed_rms"] else np.zeros(0)
    mel_rms = np.concatenate(buf["mel_rms"]) if buf["mel_rms"] else np.zeros(0)
    n = min(len(embed_rms), len(mel_rms))
    t = (np.arange(n) + 0.5) * EMBED_FRAME_S
    stacks = {}
    for key, slot in buf["stacks"].items():
        stacks[key] = {name: np.concatenate(parts) for name, parts in slot.items()}
    return {"t": t[:n], "embed_rms": embed_rms[:n], "mel_rms": mel_rms[:n], "stacks": stacks, "ds": buf["ds"]}


def _mean_mask(values, times, mask_fn) -> float | None:
    m = mask_fn(times)
    if not np.any(m):
        return None
    return float(np.mean(values[m]))


def _band_grid(feats: np.ndarray, t0: float, t1: float, n_bands: int = 8) -> dict:
    """CMVN log-mel, averaged into bands. Rows are low frequency first."""
    hop = 0.01
    a = max(0, int(t0 / hop))
    b = min(feats.shape[0], max(a + 1, int(np.ceil(t1 / hop))))
    sl = feats[a:b]
    # 4 feature frames → 40 ms, matching an encoder frame.
    n = sl.shape[0] // 4
    sl = sl[: n * 4].reshape(n, 4, -1).mean(1)
    bins = sl.shape[-1]
    edges = np.linspace(0, bins, n_bands + 1).astype(int)
    bands = np.stack([sl[:, edges[i] : edges[i + 1]].mean(-1) for i in range(n_bands)], axis=0)
    return {
        "t0": round(t0, 3),
        "t1": round(t1, 3),
        "dt": 0.04,
        "bands": [[round(float(x), 2) for x in row] for row in bands],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/zipformer_s_streaming_pool.yaml")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--wav", required=True)
    p.add_argument("--ref-srt", required=True)
    p.add_argument("--pump-ms", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = load_yaml(_resolve(args.config))
    exp_dir = _resolve(cfg["paths"]["exp_dir"])
    ckpt_path = _resolve(args.ckpt) if args.ckpt else exp_dir / "best.pt"
    out = _resolve(args.out) if args.out else exp_dir / "pipeline.json"

    wav, sr = _load_wav(_resolve(args.wav))
    ref_words = youtube_words(_resolve(args.ref_srt))
    load_icefall()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(cfg, exp_dir)
    cmvn = load_cmvn(cfg, exp_dir, compute_if_missing=False)
    model = build_zipformer(cfg, tokenizer.vocab_size)
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    model.to(device).eval()
    buf = _attach(model)

    lm_path, uni_path = _find_lm(None)
    kenlm = build_decoder(
        labels_from_tokenizer(tokenizer),
        str(lm_path),
        unigrams=load_unigrams(uni_path),
        alpha=0.5,
        beta=0.8,
    )

    with torch.inference_mode():
        sess = stream_utterance(model, cmvn, wav, sr, device, args.pump_ms)
        encoder_out = sess.encoder_out
        lens = torch.tensor([encoder_out.size(1)], device=device)
        logits = model.ctc_output[1](encoder_out)
        log_probs = F.log_softmax(logits[0], dim=-1)
        probs = log_probs.exp()
        blank = probs[:, tokenizer.blank_id].detach().float().cpu().numpy()
        kenlm_text, hyp_words = kenlm_words(kenlm, logits, int(lens[0]), 100, frame_s=FRAME_S)
        _ = greedy_decode(logits, tokenizer, lens)
        _ = greedy_search_batch(model, encoder_out, lens)

    traced = _cat(buf)
    feats = sess.feats.detach().float().cpu().numpy()
    n_enc = int(lens[0])
    enc_t = (np.arange(n_enc) + 0.5) * FRAME_S

    hyp_n = [geez_normalize(w["text"], fold_geez=True) for w in hyp_words]
    ref_n = [w["text"] for w in ref_words]
    hyp_keep = [(i, w) for i, w in enumerate(hyp_n) if w]
    ref_keep = [(i, w) for i, w in enumerate(ref_n) if w]
    ops = _align([w for _, w in ref_keep], [w for _, w in hyp_keep])

    drops = []
    ri = hi = 0
    for kind, rw, hw, _extra in ops:
        r_idx = [ref_keep[ri + k][0] for k in range(len(rw))]
        h_idx = [hyp_keep[hi + k][0] for k in range(len(hw))]
        ri += len(rw)
        hi += len(hw)
        if kind not in ("drop", "drop_tail"):
            continue
        spans = [(ref_words[i]["start_s"], ref_words[i]["end_s"]) for i in r_idx]
        spans += [(hyp_words[i]["start_s"], hyp_words[i]["end_s"]) for i in h_idx]
        if not spans:
            continue
        t0, t1 = min(s for s, _ in spans), max(e for _, e in spans)
        sl = (enc_t >= t0) & (enc_t < t1)
        if not np.any(sl):
            continue
        if float(blank[sl].mean()) < 0.7:
            continue
        drops.append({"ref": " ".join(rw), "t0": t0, "t1": max(t1, t0 + 0.12), "blank": float(blank[sl].mean())})

    # Control: encoder frames where CTC is not blank and no drop overlaps.
    drop_frame = np.zeros(n_enc, dtype=bool)
    for d in drops:
        drop_frame |= (enc_t >= d["t0"]) & (enc_t < d["t1"])
    # Letter spikes, and ordinary blanks inside words the model kept.
    # The second control checks whether a blanked drop looks like any blank.
    control_frame = (blank < 0.25) & ~drop_frame
    blank_ok = (blank >= 0.85) & ~drop_frame

    def span_mask(times, t0, t1):
        return (times >= t0) & (times < t1)

    def region_mean(times, values, frames_bool) -> float | None:
        # Map encoder-frame mask onto this series by time overlap with those frames.
        if times.size == 0 or not np.any(frames_bool):
            return None
        # A series sample counts if the nearest encoder frame is in the mask.
        idx = np.clip(np.round(times / FRAME_S - 0.5).astype(int), 0, n_enc - 1)
        m = frames_bool[idx]
        if not np.any(m):
            return None
        return float(np.mean(values[m]))

    def stack_row(stack: int) -> dict:
        layers = [traced["stacks"][k] for k in sorted(traced["stacks"]) if k[0] == stack]
        keys = ("ent", "local", "far", "lag", "src", "conv")
        regions = (("drop", drop_frame), ("ctrl", control_frame), ("blank_ok", blank_ok))
        row = {"stack": stack, "ds": traced["ds"][stack], "frame_ms": int(20 * traced["ds"][stack])}
        for key in keys:
            for name, mask in regions:
                vals = []
                for layer in layers:
                    v = region_mean(layer["t"], layer[key], mask)
                    if v is not None:
                        vals.append(v)
                row[f"{name}_{key}"] = round(float(np.mean(vals)), 3) if vals else None
        for name, mask in regions:
            hists = []
            for layer in layers:
                idx = np.clip(np.round(layer["t"] / FRAME_S - 0.5).astype(int), 0, n_enc - 1)
                m = mask[idx]
                if np.any(m):
                    hists.append(layer["hist"][m].mean(0))
            row[f"{name}_hist"] = [round(float(x), 3) for x in np.mean(hists, 0)] if hists else None
        return row

    stacks = [stack_row(i) for i in range(len(traced["ds"]))]

    def front(frames_bool) -> dict:
        idx = np.clip(np.round(traced["t"] / FRAME_S - 0.5).astype(int), 0, n_enc - 1)
        m = frames_bool[idx]
        mel = float(traced["mel_rms"][m].mean()) if np.any(m) else None
        emb = float(traced["embed_rms"][m].mean()) if np.any(m) else None
        return {
            "mel_rms": None if mel is None else round(mel, 3),
            "embed_rms": None if emb is None else round(emb, 3),
            "embed_over_mel": None if not mel else round(emb / mel, 3),
        }

    # Spectrogram slices: loudest blanked drop, and a control span of the same length.
    spec = {}
    if drops:
        worst = max(drops, key=lambda d: d["blank"])
        dur = worst["t1"] - worst["t0"]
        # Control span: longest run of control frames.
        best_t0 = None
        run = 0
        run_start = 0
        best_run = 0
        for i, flag in enumerate(control_frame):
            if flag:
                if run == 0:
                    run_start = i
                run += 1
                if run > best_run:
                    best_run = run
                    best_t0 = enc_t[run_start] - FRAME_S / 2
            else:
                run = 0
        spec["drop"] = {"label": worst["ref"], **_band_grid(feats, worst["t0"] - 0.12, worst["t1"] + 0.12)}
        if best_t0 is not None:
            spec["control"] = {
                "label": "CTC emitting",
                **_band_grid(feats, best_t0, best_t0 + dur + 0.24),
            }

    examples = []
    for d in sorted(drops, key=lambda x: -x["blank"])[:4]:
        ex = {"ref": d["ref"], "t0": round(d["t0"], 3), "t1": round(d["t1"], 3), "blank": round(d["blank"], 3)}
        m = span_mask(traced["t"], d["t0"], d["t1"])
        if np.any(m):
            ex["mel_rms"] = round(float(traced["mel_rms"][m].mean()), 3)
            ex["embed_rms"] = round(float(traced["embed_rms"][m].mean()), 3)
        # Stack 0 (20 ms) and the coarsest stack.
        for stack in (0, int(np.argmax(traced["ds"]))):
            layers = [traced["stacks"][k] for k in sorted(traced["stacks"]) if k[0] == stack]
            if not layers:
                continue
            layer = layers[0]
            mm = span_mask(layer["t"], d["t0"], d["t1"])
            if not np.any(mm):
                continue
            ex[f"s{stack}_ent"] = round(float(layer["ent"][mm].mean()), 3)
            ex[f"s{stack}_local"] = round(float(layer["local"][mm].mean()), 3)
            ex[f"s{stack}_far"] = round(float(layer["far"][mm].mean()), 3)
            ex[f"s{stack}_conv_over_src"] = round(
                float(layer["conv"][mm].mean()) / max(float(layer["src"][mm].mean()), 1e-6),
                3,
            )
        examples.append(ex)

    result = {
        "wav": str(_resolve(args.wav)),
        "ckpt": str(ckpt_path),
        "n_drops_blanked": len(drops),
        "n_control_frames": int(control_frame.sum()),
        "n_blank_ok_frames": int(blank_ok.sum()),
        "n_drop_frames": int(drop_frame.sum()),
        "front": {
            "drop": front(drop_frame),
            "control": front(control_frame),
            "blank_ok": front(blank_ok),
        },
        "stacks": stacks,
        "examples": examples,
        "spectrogram": spec,
        "lag_bins_s": ["0-0.08", "0.08-0.32", "0.32-1.28", ">=1.28"],
        "transcript_kenlm": kenlm_text,
    }
    save_json(out, result)
    log.info("wrote %s  blanked drops %d  control frames %d", out, len(drops), int(control_frame.sum()))
    for row in stacks:
        log.info(
            "stack %d ds=%d drop ent %.3f local %.3f conv/src %.3f | ctrl ent %.3f local %.3f conv/src %.3f",
            row["stack"],
            row["ds"],
            row["drop_ent"] or -1,
            row["drop_local"] or -1,
            (row["drop_conv"] or 0) / max(row["drop_src"] or 1e-6, 1e-6),
            row["ctrl_ent"] or -1,
            row["ctrl_local"] or -1,
            (row["ctrl_conv"] or 0) / max(row["ctrl_src"] or 1e-6, 1e-6),
        )
    log.info("front drop %s control %s", result["front"]["drop"], result["front"]["control"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
