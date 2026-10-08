"""Zipformer-S streaming model wired to the icefall recipe.

The modules are the files under ``third_party/icefall`` (commit recorded in
``third_party/icefall/SOURCE``). Construction follows
``egs/librispeech/ASR/zipformer/train.py``:

  get_encoder_embed, get_encoder_model, get_decoder_model,
  get_joiner_model, get_model

The loss mix follows ``compute_loss`` in that file: a simple (unpruned)
transducer loss, the pruned transducer loss, and a CTC term.
"""

from __future__ import annotations

import sys

import torch

from utils import ROOT

_ICEFALL = ROOT / "third_party" / "icefall"
_ZIPFORMER = _ICEFALL / "egs" / "librispeech" / "ASR" / "zipformer"
_LOADED = False


def load_icefall() -> None:
    """Put the vendored recipe on ``sys.path`` so its sibling imports resolve."""
    global _LOADED
    if _LOADED:
        return
    for path in (str(_ICEFALL), str(_ZIPFORMER)):
        if path not in sys.path:
            sys.path.insert(0, path)
    _LOADED = True


def _need(name: str):
    try:
        return __import__(name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"{name} is required by the icefall Zipformer recipe. "
            "k2 wheels are specific to this Python and this torch build; "
            "importing third_party/icefall/icefall/utils.py prints the matching wheel."
        ) from exc


def _ints(value) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,)
    return tuple(int(v) for v in value)


def build_zipformer(cfg: dict, vocab_size: int):
    """Build ``AsrModel`` the way ``get_model`` does.

    ``blank_id`` is 0. Icefall's LibriSpeech run uses ``<blk>``, which
    sentencepiece puts at index 0; this repo's character tokenizer does the
    same with ``<blank>``.
    """
    load_icefall()
    _need("k2")
    _need("lhotse")
    from model import AsrModel
    from subsampling import Conv2dSubsampling
    from decoder import Decoder
    from joiner import Joiner
    from scaling import ScheduledFloat
    from zipformer import Zipformer2

    m = cfg["model"]
    blank_id = int(m.get("blank_id", 0))
    if blank_id != 0:
        raise ValueError("the icefall joiner and greedy decode treat blank as index 0")
    encoder_dim = _ints(m["encoder_dim"])
    causal = bool(m.get("causal", True))
    chunk_size = _ints(m.get("chunk_size", [16, 32, 64, -1]))
    if not causal and chunk_size != (-1,):
        raise ValueError("icefall requires chunk_size [-1] when causal is false")

    encoder_embed = Conv2dSubsampling(
        in_channels=int(m.get("input_dim", 80)),
        out_channels=encoder_dim[0],
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
    )
    encoder = Zipformer2(
        output_downsampling_factor=2,
        downsampling_factor=_ints(m["downsampling_factor"]),
        num_encoder_layers=_ints(m["num_encoder_layers"]),
        encoder_dim=encoder_dim,
        encoder_unmasked_dim=_ints(m["encoder_unmasked_dim"]),
        query_head_dim=_ints(m.get("query_head_dim", 32)),
        pos_head_dim=_ints(m.get("pos_head_dim", 4)),
        value_head_dim=_ints(m.get("value_head_dim", 12)),
        pos_dim=int(m.get("pos_dim", 48)),
        num_heads=_ints(m["num_heads"]),
        feedforward_dim=_ints(m["feedforward_dim"]),
        cnn_module_kernel=_ints(m["cnn_module_kernel"]),
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches=4000.0,
        causal=causal,
        chunk_size=chunk_size,
        left_context_frames=_ints(m.get("left_context_frames", [128, 256, -1])),
    )
    decoder_dim = int(m.get("decoder_dim", 512))
    decoder = Decoder(
        vocab_size=vocab_size,
        decoder_dim=decoder_dim,
        blank_id=blank_id,
        context_size=int(m.get("context_size", 2)),
    )
    joiner = Joiner(
        encoder_dim=max(encoder_dim),
        decoder_dim=decoder_dim,
        joiner_dim=int(m.get("joiner_dim", 512)),
        vocab_size=vocab_size,
    )
    model = AsrModel(
        encoder_embed=encoder_embed,
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        attention_decoder=None,
        encoder_dim=max(encoder_dim),
        decoder_dim=decoder_dim,
        vocab_size=vocab_size,
        use_transducer=True,
        use_ctc=True,
        use_attention_decoder=False,
    )
    return model


def scaled_adam(model, base_lr: float):
    """``ScaledAdam`` as constructed in ``zipformer/train.py``."""
    load_icefall()
    from icefall.utils import get_parameter_groups_with_lrs
    from optim import ScaledAdam

    return ScaledAdam(
        get_parameter_groups_with_lrs(model, lr=base_lr, include_names=True),
        lr=base_lr,
        clipping_scale=2.0,
    )


def eden(optimizer, tcfg: dict):
    """Eden scheduler. ``train.py`` passes ``warmup_start=0.1`` and leaves
    ``warmup_batches`` at the class default of 500.
    """
    load_icefall()
    from optim import Eden

    return Eden(
        optimizer,
        float(tcfg.get("lr_batches", 7500)),
        float(tcfg.get("lr_epochs", 3.5)),
        warmup_batches=float(tcfg.get("warmup_batches", 500)),
        warmup_start=float(tcfg.get("warmup_start", 0.1)),
    )


def loss_scales(batch_idx: int, tcfg: dict) -> tuple[float, float, float]:
    """Simple, pruned, and CTC scales from ``compute_loss``.

    ``batch_idx`` is icefall's ``batch_idx_train`` after it has been
    incremented for the current step (the first update is 1). Before
    ``warm_step`` the simple term is annealed from 1 down to
    ``simple_loss_scale`` and the pruned term is annealed from 0.1 up to 1.
    ``warm_step`` of 0 keeps the final scales, which is the pool fine-tune.
    """
    warm = int(tcfg.get("warm_step", 2000))
    simple_final = float(tcfg.get("simple_loss_scale", 0.5))
    ctc_scale = float(tcfg.get("ctc_loss_scale", 0.2))
    if warm <= 0 or batch_idx >= warm:
        return simple_final, 1.0, ctc_scale
    frac = batch_idx / warm
    simple = 1.0 - frac * (1.0 - simple_final)
    pruned = 0.1 + 0.9 * frac
    return simple, pruned, ctc_scale


def combine_loss(
    simple_loss: torch.Tensor,
    pruned_loss: torch.Tensor,
    ctc_loss: torch.Tensor,
    *,
    batch_idx: int,
    tcfg: dict,
) -> torch.Tensor:
    simple_s, pruned_s, ctc_s = loss_scales(batch_idx, tcfg)
    # CTC is computed from CPU targets inside AsrModel.forward_ctc. The sum
    # has to sit on the same device as the k2 transducer losses.
    return simple_s * simple_loss + pruned_s * pruned_loss + ctc_s * ctc_loss.to(simple_loss.device)


def encoder_frames(mel_frames: int) -> int:
    """Joiner/CTC length. ``zipformer/train.py`` ``remove_short_and_long_utt``:

    ``T = ((num_frames - 7) // 2 + 1) // 2``

    which is Conv2dSubsampling ``(T - 7) // 2`` then Zipformer2's output
    downsample ``(x_lens + 1) // 2``.
    """
    return ((int(mel_frames) - 7) // 2 + 1) // 2


def labels_ok(mel_frames: int, n_tokens: int) -> bool:
    """CTC needs ``T - 2 >=`` tokens. The same check is in ``train.py``."""
    return encoder_frames(mel_frames) - 2 >= int(n_tokens)


def ragged_labels(targets: torch.Tensor, target_lens: torch.Tensor):
    """``k2.RaggedTensor`` of shape [utt][token], as ``compute_loss`` builds."""
    load_icefall()
    import k2

    # k2.RaggedTensor takes a list of token-id lists, which is what
    # sentencepiece produces in zipformer/train.py compute_loss.
    rows = []
    lens = target_lens.detach().cpu().tolist()
    cpu = targets.detach().cpu()
    for i, n in enumerate(lens):
        rows.append([int(v) for v in cpu[i, : int(n)].tolist()])
    return k2.RaggedTensor(rows)
