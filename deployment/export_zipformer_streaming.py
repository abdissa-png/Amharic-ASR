"""ONNX export of a streaming Zipformer-S transducer.

The graphs match the live cache in ``decoding.zipformer_stream`` and the
icefall streaming export (encoder / decoder / joiner, chunk 32, left 128).
The encoder also returns CTC logits from the linear head the demo rescores.
Features are log-Mel after CMVN, 77 frames per call (64 new, 13 of right context).

    python -m deployment.export_zipformer_streaming \\
        --config configs/zipformer_s_streaming_pool_leyu.yaml \\
        --ckpt exp/zipformer_s_streaming_pool_leyu/last.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from decoding.zipformer_greedy import greedy_search_continue, initial_context
from models.features import SAMPLE_RATE, log_mel
from decoding.zipformer_stream import (
    EMBED_CHUNK,
    FEATURE_CHUNK,
    FEATURE_SHIFT,
    init_stream_states,
    stream_encoder_chunk,
)
from models.zipformer_icefall import build_zipformer, load_icefall
from train import load_tokenizer
from utils import ROOT, load_yaml, setup_logging

log = setup_logging()

LEFT_CONTEXT = 128


def _strip_train_only(model: nn.Module) -> int:
    """Drop modules that are identity at eval and do not export."""
    load_icefall()
    from scaling import Balancer, Dropout3, ScaleGrad, Whiten

    names = [
        name
        for name, mod in model.named_modules()
        if isinstance(mod, (Balancer, Dropout3, ScaleGrad, Whiten))
    ]
    for name in names:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, nn.Identity())
    return len(names)


def _load(cfg: dict, ckpt: Path) -> tuple[nn.Module, int]:
    exp_dir = ROOT / cfg["paths"]["exp_dir"]
    tokenizer = load_tokenizer(cfg, exp_dir)
    model = build_zipformer(cfg, tokenizer.vocab_size)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, int(blob.get("epoch", -1))


class OnnxEncoder(nn.Module):
    """One cached chunk. ``encoder_out`` is already the joiner projection."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.encoder_proj = model.joiner.encoder_proj
        self.ctc_linear = model.ctc_output[1]

    def forward(self, x: torch.Tensor, *states: torch.Tensor):
        encoder_out, new_states = stream_encoder_chunk(self.model, x, list(states))
        projected = self.encoder_proj(encoder_out)
        ctc_logits = self.ctc_linear(encoder_out)
        return (projected, *new_states, ctc_logits)


class OnnxDecoder(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.decoder = model.decoder
        self.decoder_proj = model.joiner.decoder_proj

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # need_pad is false at inference: y is already context_size tokens.
        out = self.decoder(y, need_pad=False).squeeze(1)
        return self.decoder_proj(out)


class OnnxJoiner(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.output_linear = model.joiner.output_linear

    def forward(self, encoder_out: torch.Tensor, decoder_out: torch.Tensor) -> torch.Tensor:
        # Both inputs are already projected, one frame each: (N, joiner_dim).
        return self.output_linear(torch.tanh(encoder_out + decoder_out))


def _state_names(n_layers: int) -> list[str]:
    names = []
    for i in range(n_layers):
        names.extend(
            [
                f"cached_key_{i}",
                f"cached_nonlin_attn_{i}",
                f"cached_val1_{i}",
                f"cached_val2_{i}",
                f"cached_conv1_{i}",
                f"cached_conv2_{i}",
            ]
        )
    names.append("embed_states")
    names.append("processed_lens")
    return names


def _add_meta(path: Path, meta: dict[str, str]) -> None:
    import onnx

    model = onnx.load(str(path))
    del model.metadata_props[:]
    for key, value in meta.items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = str(value)
    onnx.save(model, str(path))


def _export(model: nn.Module, args, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(model, args, str(path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, args, str(path), **kwargs)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _write_tokens(path: Path, itos: list[str]) -> None:
    lines = []
    for i, tok in enumerate(itos):
        # The space piece is a leading space so a reader that splits on the
        # last space keeps the symbol " ".
        if tok == " ":
            lines.append(f"  {i}")
        else:
            lines.append(f"{tok} {i}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_encoder(sess, feats: np.ndarray, states: list[np.ndarray]):
    feed = {"x": feats}
    for name, value in zip(_state_names((len(states) - 2) // 6), states):
        feed[name] = value
    out = sess.run(None, feed)
    n_states = len(states)
    projected = out[0]
    new_states = out[1 : 1 + n_states]
    ctc = out[1 + n_states]
    return projected, new_states, ctc


def verify(
    ref: nn.Module,
    enc_path: Path,
    dec_path: Path,
    join_path: Path,
    *,
    chunk: int,
    left: int,
) -> dict:
    import onnxruntime as ort

    if int(ref.encoder.left_context_frames[0]) != left:
        raise RuntimeError(f"model left context is {ref.encoder.left_context_frames[0]}, export used {left}")
    if chunk != EMBED_CHUNK:
        raise RuntimeError(f"this streaming cache is chunk {EMBED_CHUNK}, not {chunk}")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    enc = ort.InferenceSession(str(enc_path), opts, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(dec_path), opts, providers=["CPUExecutionProvider"])
    join = ort.InferenceSession(str(join_path), opts, providers=["CPUExecutionProvider"])

    g = torch.Generator().manual_seed(0)
    chunks = [torch.randn(1, FEATURE_CHUNK, 80, generator=g) for _ in range(3)]
    pt_states = init_stream_states(ref, torch.device("cpu"))
    np_states = [t.detach().cpu().numpy() for t in pt_states]
    context = initial_context(ref)
    pt_tokens: list[int] = []
    ox_tokens: list[int] = []
    blank = int(ref.decoder.blank_id)
    unk = int(getattr(ref, "unk_id", blank))
    report = []

    for step, feats in enumerate(chunks):
        with torch.no_grad():
            enc_out, pt_states = stream_encoder_chunk(ref, feats, pt_states)
            projected = ref.joiner.encoder_proj(enc_out)
            ctc = ref.ctc_output[1](enc_out)
            tokens, context = greedy_search_continue(ref, enc_out, context)
        pt_tokens.extend(tokens)

        ox_proj, np_states, ox_ctc = _run_encoder(enc, feats.numpy(), np_states)
        row = {
            "chunk": step,
            "encoder_cosine": _cosine(projected.numpy(), ox_proj),
            "encoder_max_abs": _max_abs(projected.numpy(), ox_proj),
            "ctc_cosine": _cosine(ctc.numpy(), ox_ctc),
            "ctc_max_abs": _max_abs(ctc.numpy(), ox_ctc),
        }
        state_abs = [
            _max_abs(a.numpy(), b) for a, b in zip(pt_states, np_states)
        ]
        row["state_max_abs"] = max(state_abs)
        report.append(row)
        log.info(
            "chunk %d  encoder cosine=%.6f max_abs=%.4g  ctc cosine=%.6f max_abs=%.4g  state max_abs=%.4g",
            step,
            row["encoder_cosine"],
            row["encoder_max_abs"],
            row["ctc_cosine"],
            row["ctc_max_abs"],
            row["state_max_abs"],
        )

    # Independent ONNX greedy over the same chunks.
    np_states = [t.detach().cpu().numpy() for t in init_stream_states(ref, torch.device("cpu"))]
    ctx_np = initial_context(ref).numpy()
    for feats in chunks:
        ox_proj, np_states, _ctc = _run_encoder(enc, feats.numpy(), np_states)
        for t in range(ox_proj.shape[1]):
            dec_out = dec.run(None, {"y": ctx_np})[0]
            logits = join.run(
                None,
                {
                    "encoder_out": np.ascontiguousarray(ox_proj[:, t, :]),
                    "decoder_out": dec_out,
                },
            )[0]
            y = int(np.argmax(logits, axis=-1)[0])
            if y != blank and y != unk:
                ox_tokens.append(y)
                ctx_np = np.concatenate([ctx_np[:, 1:], np.array([[y]], dtype=np.int64)], axis=1)

    # Decoder and joiner against the PyTorch modules on a negative context.
    y = initial_context(ref)
    with torch.no_grad():
        pt_dec = ref.joiner.decoder_proj(ref.decoder(y, need_pad=False)).squeeze(1)
    ox_dec = dec.run(None, {"y": y.numpy()})[0]
    enc_vec = torch.randn(1, int(pt_dec.shape[-1]))
    with torch.no_grad():
        pt_join = ref.joiner.output_linear(torch.tanh(enc_vec + pt_dec))
    ox_join = join.run(
        None,
        {"encoder_out": enc_vec.numpy(), "decoder_out": ox_dec},
    )[0]
    head = {
        "decoder_cosine": _cosine(pt_dec.numpy(), ox_dec),
        "decoder_max_abs": _max_abs(pt_dec.numpy(), ox_dec),
        "joiner_cosine": _cosine(pt_join.numpy(), ox_join),
        "joiner_max_abs": _max_abs(pt_join.numpy(), ox_join),
        "pytorch_tokens": pt_tokens,
        "onnx_tokens": ox_tokens,
        "tokens_match": pt_tokens == ox_tokens,
        "chunks": report,
    }
    log.info(
        "decoder cosine=%.6f  joiner cosine=%.6f  tokens pytorch=%s onnx=%s",
        head["decoder_cosine"],
        head["joiner_cosine"],
        pt_tokens,
        ox_tokens,
    )
    worst_enc = min(row["encoder_cosine"] for row in report)
    worst_ctc = min(row["ctc_cosine"] for row in report)
    if worst_enc < 0.999 or worst_ctc < 0.999 or not head["tokens_match"]:
        raise RuntimeError(
            f"ONNX mismatch encoder={worst_enc:.6f} ctc={worst_ctc:.6f} tokens_match={head['tokens_match']}"
        )
    if head["decoder_cosine"] < 0.999 or head["joiner_cosine"] < 0.999:
        raise RuntimeError("decoder or joiner ONNX mismatch")
    speech = _speech_tokens(ref, enc, dec, join)
    if speech is not None:
        head["speech"] = speech
        if not speech["tokens_match"] or not speech["pytorch_tokens"]:
            raise RuntimeError(f"speech token mismatch {speech}")
    return head


def _speech_tokens(ref, enc, dec, join) -> dict | None:
    """Greedy tokens on a few seconds of the EBC clip, PyTorch against ONNX."""
    wav_path = ROOT / "exp/zipformer_s_streaming_pool/youtube_timed/YiB33Piqh20.wav"
    cmvn_path = ROOT / "exp/zipformer_s_streaming_pool_leyu/cmvn.pt"
    if not wav_path.exists() or not cmvn_path.exists():
        return None
    import torchaudio
    from models.features import GlobalCMVN

    wav, sr = torchaudio.load(str(wav_path))
    wav = wav.mean(0)[: int(5 * SAMPLE_RATE)]
    if int(sr) != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), int(sr), SAMPLE_RATE).squeeze(0)
    feats = GlobalCMVN.load(cmvn_path)(log_mel(wav, dither=0.0))
    blank = int(ref.decoder.blank_id)
    unk = int(getattr(ref, "unk_id", blank))
    pt_states = init_stream_states(ref, torch.device("cpu"))
    np_states = [t.detach().cpu().numpy() for t in pt_states]
    context = initial_context(ref)
    ctx_np = context.numpy().copy()
    pt_tokens: list[int] = []
    ox_tokens: list[int] = []
    done = 0
    while done + FEATURE_CHUNK <= int(feats.size(0)):
        chunk = feats[done : done + FEATURE_CHUNK].unsqueeze(0)
        with torch.no_grad():
            enc_out, pt_states = stream_encoder_chunk(ref, chunk, pt_states)
            tokens, context = greedy_search_continue(ref, enc_out, context)
        pt_tokens.extend(tokens)
        ox_proj, np_states, _ctc = _run_encoder(enc, chunk.numpy(), np_states)
        for t in range(ox_proj.shape[1]):
            dec_out = dec.run(None, {"y": ctx_np})[0]
            logits = join.run(
                None,
                {
                    "encoder_out": np.ascontiguousarray(ox_proj[:, t, :]),
                    "decoder_out": dec_out,
                },
            )[0]
            y = int(np.argmax(logits, axis=-1)[0])
            if y != blank and y != unk:
                ox_tokens.append(y)
                ctx_np = np.concatenate([ctx_np[:, 1:], np.array([[y]], dtype=np.int64)], axis=1)
        done += FEATURE_SHIFT
    log.info("speech tokens pytorch=%s onnx=%s", pt_tokens, ox_tokens)
    return {"pytorch_tokens": pt_tokens, "onnx_tokens": ox_tokens, "tokens_match": pt_tokens == ox_tokens}


def _quantize(src: Path, dest: Path, op_types: list[str], meta: dict[str, str]) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(src),
        model_output=str(dest),
        op_types_to_quantize=op_types,
        weight_type=QuantType.QInt8,
    )
    _add_meta(dest, meta)
    log.info("wrote %s (%.1f MB)", dest, dest.stat().st_size / 1e6)


def export(config: Path, ckpt: Path, out_dir: Path, *, chunk: int, left: int, opset: int, quantize: bool) -> Path:
    if chunk != EMBED_CHUNK or left != LEFT_CONTEXT:
        raise ValueError(
            f"the live cache is chunk {EMBED_CHUNK} and left context {LEFT_CONTEXT}, not {chunk} and {left}"
        )
    cfg = load_yaml(config)
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    ref, epoch = _load(cfg, ckpt)
    model, _epoch = _load(cfg, ckpt)
    n_id = _strip_train_only(model)
    log.info("replaced %d train-only modules with Identity", n_id)

    exp_dir = ROOT / cfg["paths"]["exp_dir"]
    tokenizer = load_tokenizer(cfg, exp_dir)
    encoder = OnnxEncoder(model).eval()
    decoder = OnnxDecoder(model).eval()
    joiner = OnnxJoiner(model).eval()

    states = init_stream_states(model, torch.device("cpu"))
    n_layers = (len(states) - 2) // 6
    state_names = _state_names(n_layers)
    x = torch.randn(1, FEATURE_CHUNK, 80)
    if len(state_names) != len(states):
        raise RuntimeError(f"{len(state_names)} names for {len(states)} states")

    out_dir.mkdir(parents=True, exist_ok=True)
    enc_path = out_dir / f"encoder-chunk{chunk}-left{left}.onnx"
    dec_path = out_dir / f"decoder-chunk{chunk}-left{left}.onnx"
    join_path = out_dir / f"joiner-chunk{chunk}-left{left}.onnx"

    dyn_in = {"x": {0: "N"}}
    dyn_out = {"encoder_out": {0: "N"}, "ctc_logits": {0: "N"}}
    for name, tensor in zip(state_names, states):
        # batch axis: conv caches and embed states use dim 0; attention caches use dim 1
        # except cached_nonlin_attn, which is (1, N, left, dim)
        if tensor.ndim >= 1 and tensor.shape[0] == 1 and name.startswith("cached_nonlin"):
            axis = 1
        elif name.startswith("cached_key") or name.startswith("cached_val"):
            axis = 1
        else:
            axis = 0
        dyn_in[name] = {axis: "N"}
        dyn_out[f"new_{name}" if name != "embed_states" else "new_embed_states"] = {axis: "N"}
        if name == "processed_lens":
            dyn_out["new_processed_lens"] = {0: "N"}

    new_state_names = []
    for name in state_names:
        if name == "embed_states":
            new_state_names.append("new_embed_states")
        elif name == "processed_lens":
            new_state_names.append("new_processed_lens")
        else:
            new_state_names.append(f"new_{name}")

    log.info("exporting encoder  states=%d  T=%d  layers=%d", len(states), FEATURE_CHUNK, n_layers)
    _export(
        encoder,
        (x, *states),
        enc_path,
        input_names=["x", *state_names],
        output_names=["encoder_out", *new_state_names, "ctc_logits"],
        dynamic_axes={**dyn_in, **dyn_out},
        opset_version=opset,
        do_constant_folding=True,
    )
    ds = [int(v) for v in model.encoder.downsampling_factor]
    enc_meta = {
        "model_type": "zipformer2",
        "version": "1",
        "comment": "streaming zipformer2",
        "decode_chunk_len": str(FEATURE_SHIFT),
        "T": str(FEATURE_CHUNK),
        "chunk_size": str(chunk),
        "left_context_frames": str(left),
        "num_encoder_layers": ",".join(str(int(v)) for v in model.encoder.num_encoder_layers),
        "encoder_dims": ",".join(str(int(v)) for v in model.encoder.encoder_dim),
        "cnn_module_kernels": ",".join(str(int(v)) for v in model.encoder.cnn_module_kernel),
        "left_context_len": ",".join(str(left // k) for k in ds),
        "query_head_dims": ",".join(str(int(v)) for v in model.encoder.query_head_dim),
        "value_head_dims": ",".join(str(int(v)) for v in model.encoder.value_head_dim),
        "num_heads": ",".join(str(int(v)) for v in model.encoder.num_heads),
        "vocab_size": str(tokenizer.vocab_size),
        "epoch": str(epoch),
        "ctc_output": "ctc_logits",
    }
    _add_meta(enc_path, enc_meta)
    log.info("wrote %s (%.1f MB)", enc_path, enc_path.stat().st_size / 1e6)

    y = torch.zeros(1, int(model.decoder.context_size), dtype=torch.int64)
    _export(
        decoder,
        y,
        dec_path,
        input_names=["y"],
        output_names=["decoder_out"],
        dynamic_axes={"y": {0: "N"}, "decoder_out": {0: "N"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    dec_meta = {
        "context_size": str(int(model.decoder.context_size)),
        "vocab_size": str(tokenizer.vocab_size),
    }
    _add_meta(dec_path, dec_meta)
    log.info("wrote %s (%.1f MB)", dec_path, dec_path.stat().st_size / 1e6)

    joiner_dim = int(joiner.output_linear.in_features)
    _export(
        joiner,
        (torch.randn(1, joiner_dim), torch.randn(1, joiner_dim)),
        join_path,
        input_names=["encoder_out", "decoder_out"],
        output_names=["logit"],
        dynamic_axes={
            "encoder_out": {0: "N"},
            "decoder_out": {0: "N"},
            "logit": {0: "N"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    join_meta = {"joiner_dim": str(joiner_dim)}
    _add_meta(join_path, join_meta)
    log.info("wrote %s (%.1f MB)", join_path, join_path.stat().st_size / 1e6)

    _write_tokens(out_dir / "tokens.txt", list(tokenizer.itos))
    report = verify(ref, enc_path, dec_path, join_path, chunk=chunk, left=left)

    if quantize:
        _quantize(enc_path, out_dir / f"encoder-chunk{chunk}-left{left}.int8.onnx", ["MatMul"], enc_meta)
        _quantize(dec_path, out_dir / f"decoder-chunk{chunk}-left{left}.int8.onnx", ["MatMul", "Gather"], dec_meta)
        _quantize(join_path, out_dir / f"joiner-chunk{chunk}-left{left}.int8.onnx", ["MatMul"], join_meta)

    meta = {
        "checkpoint": str(ckpt),
        "epoch": epoch,
        "chunk_size": chunk,
        "left_context_frames": left,
        "feature_chunk": FEATURE_CHUNK,
        "feature_shift": FEATURE_SHIFT,
        "vocab_size": tokenizer.vocab_size,
        "joiner_dim": joiner_dim,
        "encoder": enc_path.name,
        "decoder": dec_path.name,
        "joiner": join_path.name,
        "ctc_output": "ctc_logits",
        "verify": {
            "tokens_match": report["tokens_match"],
            "encoder_cosine": [row["encoder_cosine"] for row in report["chunks"]],
            "ctc_cosine": [row["ctc_cosine"] for row in report["chunks"]],
            "speech_tokens_match": (report.get("speech") or {}).get("tokens_match"),
            "speech_tokens": (report.get("speech") or {}).get("pytorch_tokens"),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out_dir


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/zipformer_s_streaming_pool_leyu.yaml"))
    p.add_argument("--ckpt", type=Path, default=Path("exp/zipformer_s_streaming_pool_leyu/last.pt"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--chunk", type=int, default=EMBED_CHUNK)
    p.add_argument("--left", type=int, default=LEFT_CONTEXT)
    p.add_argument("--opset", type=int, default=14)
    p.add_argument("--no-int8", action="store_true")
    args = p.parse_args()
    cfg = load_yaml(args.config)
    out_dir = args.out_dir or (ROOT / cfg["paths"]["exp_dir"] / "onnx")
    export(
        args.config,
        args.ckpt,
        out_dir,
        chunk=args.chunk,
        left=args.left,
        opset=args.opset,
        quantize=not args.no_int8,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
