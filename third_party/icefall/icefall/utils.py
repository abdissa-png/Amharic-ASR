# Functions copied verbatim from icefall/utils.py at commit
# 3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1
# https://github.com/k2-fsa/icefall/blob/3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1/icefall/utils.py
#
# Zipformer2, AsrModel, and the training loop import these names. The rest of
# icefall/utils.py is not used by this pipeline and pulls in kaldialign,
# pypinyin, and tensorboard.
#
# Apache-2.0, Xiaomi Corporation / Johns Hopkins University. See ../LICENSE.

from __future__ import annotations

import logging
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from packaging import version


def _import_k2():
    """Import k2, or raise with the wheel that matches this interpreter.

    k2 wheels are tagged with the Python ABI, the torch version, and the CUDA
    version they were built against. A cp311 / torch 2.7.1 / cuda 12.6 wheel
    is rejected by pip on any other combination.
    """
    try:
        import k2
    except ModuleNotFoundError:
        pass
    else:
        return k2

    import platform
    import re
    import sys
    import urllib.request

    cp = f"cp{sys.version_info.major}{sys.version_info.minor}"
    torch_ver = torch.__version__.split("+")[0]
    cuda = torch.version.cuda or ""
    machine = platform.machine()
    arch = "aarch64" if machine in ("aarch64", "arm64") else "x86_64"
    page = "https://k2-fsa.github.io/k2/cuda.html"
    hint = (
        f"k2 is not installed. This process is CPython {sys.version_info.major}.{sys.version_info.minor}"
        f" ({cp}), torch {torch.__version__}, CUDA {cuda or 'none'}, {machine}.\n"
        "A wheel built for a different Python is rejected as 'not a supported wheel'.\n"
    )
    try:
        html = urllib.request.urlopen(page, timeout=60).read().decode("utf-8", "replace")
    except Exception as exc:
        raise ModuleNotFoundError(
            hint + f"Could not read {page} ({exc}). Open that page and install the wheel "
            f"whose name contains cuda{cuda}.torch{torch_ver}-{cp}- and {arch}."
        ) from None
    links = re.findall(r"https://huggingface.co/[^\"']+?\.whl", html)
    needle_torch = f"torch{torch_ver}"
    needle_cuda = f"cuda{cuda}."
    hits = [
        url
        for url in links
        if needle_torch in url and needle_cuda in url and f"-{cp}-" in url and arch in url
    ]
    if not hits:
        raise ModuleNotFoundError(
            hint + f"No published k2 wheel matches cuda{cuda} + torch {torch_ver} + {cp} + {arch}. "
            f"The index is {page}."
        ) from None

    def _date(url: str) -> str:
        found = re.findall(r"dev(\d{8})", url)
        return found[-1] if found else ""

    url = max(hits, key=_date)
    raise ModuleNotFoundError(
        hint + "Install this build, then rerun training:\n\n"
        f"python -m pip install lhotse {url}\n"
    ) from None


k2 = _import_k2()


# --- icefall/utils.py lines 56-56 ---
TORCH_VERSION = version.parse(torch.__version__)

# --- icefall/utils.py lines 59-97 ---
def create_grad_scaler(device="cuda", **kwargs):
    """
    Creates a GradScaler compatible with both torch < 2.3.0 and >= 2.3.0.
    Accepts all kwargs like: enabled, init_scale, growth_factor, etc.

    /icefall/egs/librispeech/ASR/./zipformer/train.py:1451: FutureWarning:
    `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use
    `torch.amp.GradScaler('cuda', args...)` instead.
    """
    if TORCH_VERSION >= version.parse("2.3.0"):
        from torch.amp import GradScaler

        return GradScaler(device=device, **kwargs)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            return torch.cuda.amp.GradScaler(**kwargs)


@contextmanager
def torch_autocast(device_type="cuda", **kwargs):
    """
    To fix the following warnings:
    /icefall/egs/librispeech/ASR/zipformer/model.py:323:
    FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated.
    Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(enabled=False):
    """
    if TORCH_VERSION >= version.parse("2.3.0"):
        # Use new unified API
        with torch.amp.autocast(device_type=device_type, **kwargs):
            yield
    else:
        # Suppress deprecation warning and use old CUDA-specific autocast
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            with torch.cuda.amp.autocast(**kwargs):
                yield


# --- icefall/utils.py lines 1353-1424 ---
def concat(ragged: k2.RaggedTensor, value: int, direction: str) -> k2.RaggedTensor:
    """Prepend a value to the beginning of each sublist or append a value.
    to the end of each sublist.

    Args:
      ragged:
        A ragged tensor with two axes.
      value:
        The value to prepend or append.
      direction:
        It can be either "left" or "right". If it is "left", we
        prepend the value to the beginning of each sublist;
        if it is "right", we append the value to the end of each
        sublist.

    Returns:
      Return a new ragged tensor, whose sublists either start with
      or end with the given value.

    >>> a = k2.RaggedTensor([[1, 3], [5]])
    >>> a
    [ [ 1 3 ] [ 5 ] ]
    >>> concat(a, value=0, direction="left")
    [ [ 0 1 3 ] [ 0 5 ] ]
    >>> concat(a, value=0, direction="right")
    [ [ 1 3 0 ] [ 5 0 ] ]

    """
    dtype = ragged.dtype
    device = ragged.device

    assert ragged.num_axes == 2, f"num_axes: {ragged.num_axes}"
    pad_values = torch.full(
        size=(ragged.tot_size(0), 1),
        fill_value=value,
        device=device,
        dtype=dtype,
    )
    pad = k2.RaggedTensor(pad_values)

    if direction == "left":
        ans = k2.ragged.cat([pad, ragged], axis=1)
    elif direction == "right":
        ans = k2.ragged.cat([ragged, pad], axis=1)
    else:
        raise ValueError(
            f'Unsupported direction: {direction}. " \
            "Expect either "left" or "right"'
        )
    return ans


def add_sos(ragged: k2.RaggedTensor, sos_id: int) -> k2.RaggedTensor:
    """Add SOS to each sublist.

    Args:
      ragged:
        A ragged tensor with two axes.
      sos_id:
        The ID of the SOS symbol.

    Returns:
      Return a new ragged tensor, where each sublist starts with SOS.

    >>> a = k2.RaggedTensor([[1, 3], [5]])
    >>> a
    [ [ 1 3 ] [ 5 ] ]
    >>> add_sos(a, sos_id=0)
    [ [ 0 1 3 ] [ 0 5 ] ]

    """
    return concat(ragged, sos_id, direction="left")

# --- icefall/utils.py lines 1449-1486 ---
def make_pad_mask(
    lengths: torch.Tensor,
    max_len: int = 0,
    pad_left: bool = False,
) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
      pad_left:
        If ``False`` (default), padding is on the right.
        If ``True``, padding is on the left.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    # A caller that passes max_len (the padded width) must cover every length;
    # reading lengths.max() would sync on a CUDA tensor.
    if max_len <= 0:
        max_len = int(lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expanded_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    if pad_left:
        mask = expanded_lengths < (max_len - lengths).unsqueeze(1)
    else:
        mask = expanded_lengths >= lengths.unsqueeze(-1)

    return mask

# --- icefall/utils.py lines 1583-1652 ---
def get_parameter_groups_with_lrs(
    model: nn.Module,
    lr: float,
    include_names: bool = False,
    freeze_modules: List[str] = [],
) -> List[dict]:
    """
    This is for use with the ScaledAdam optimizers (more recent versions that accept lists of
    named-parameters; we can, if needed, create a version without the names).

    It provides a way to specify learning-rate scales inside the module, so that if
    any nn.Module in the hierarchy has a floating-point parameter 'lr_scale', it will
    scale the LR of any parameters inside that module or its submodules.  Note: you
    can set module parameters outside the __init__ function, e.g.:
      >>> a = nn.Linear(10, 10)
      >>> a.lr_scale = 0.5

    Returns: a list of dicts, of the following form:
      if include_names == False:
        [  { 'params': [ tensor1, tensor2, ... ], 'lr': 0.01 },
           { 'params': [ tensor3, tensor4, ... ], 'lr': 0.005 },
         ...   ]
      if include_names == true:
        [  { 'named_params': [ (name1, tensor1, (name2, tensor2), ... ], 'lr': 0.01 },
           { 'named_params': [ (name3, tensor3), (name4, tensor4), ... ], 'lr': 0.005 },
         ...   ]

    """
    named_modules = list(model.named_modules())

    # flat_lr_scale just contains the lr_scale explicitly specified
    # for each prefix of the name, e.g. 'encoder.layers.3', these need
    # to be multiplied for all prefix of the name of any given parameter.
    flat_lr_scale = defaultdict(lambda: 1.0)
    names = []
    for name, m in model.named_modules():
        names.append(name)
        if hasattr(m, "lr_scale"):
            flat_lr_scale[name] = m.lr_scale

    # lr_to_parames is a dict from learning rate (floating point) to: if
    # include_names == true, a list of (name, parameter) for that learning rate;
    # otherwise a list of parameters for that learning rate.
    lr_to_params = defaultdict(list)

    for name, parameter in model.named_parameters():
        split_name = name.split(".")
        # caution: as a special case, if the name is '', split_name will be [ '' ].
        prefix = split_name[0]
        if prefix == "module":  # DDP
            module_name = split_name[1]
            if module_name in freeze_modules:
                logging.info(f"Remove {name} from parameters")
                continue
        else:
            if prefix in freeze_modules:
                logging.info(f"Remove {name} from parameters")
                continue
        cur_lr = lr * flat_lr_scale[prefix]
        if prefix != "":
            cur_lr *= flat_lr_scale[""]
        for part in split_name[1:]:
            prefix = ".".join([prefix, part])
            cur_lr *= flat_lr_scale[prefix]
        lr_to_params[cur_lr].append((name, parameter) if include_names else parameter)

    if include_names:
        return [{"named_params": pairs, "lr": lr} for lr, pairs in lr_to_params.items()]
    else:
        return [{"params": params, "lr": lr} for lr, params in lr_to_params.items()]

# --- icefall/utils.py lines 2426-2459 ---
def time_warp(
    features: torch.Tensor,
    p: float = 0.9,
    time_warp_factor: Optional[int] = 80,
    supervision_segments: Optional[torch.Tensor] = None,
):
    """Apply time warping on a batch of features"""
    # icefall/utils.py imports this at module level. Kept local because the
    # standard transducer run (use_cr_ctc=False) never calls time_warp.
    from lhotse.dataset.signal_transforms import time_warp as time_warp_impl
    if time_warp_factor is None or time_warp_factor < 1:
        return features
    assert (
        len(features.shape) == 3
    ), f"SpecAugment only supports batches of single-channel feature matrices. {features.shape}"
    features = features.clone()
    if supervision_segments is None:
        # No supervisions - apply spec augment to full feature matrices.
        for sequence_idx in range(features.size(0)):
            if random.random() > p:
                # Randomly choose whether this transform is applied
                continue
            features[sequence_idx] = time_warp_impl(
                features[sequence_idx], factor=time_warp_factor
            )
    else:
        # Supervisions provided - we will apply time warping only on the supervised areas.
        for sequence_idx, start_frame, num_frames in supervision_segments:
            if random.random() > p:
                # Randomly choose whether this transform is applied
                continue
            end_frame = start_frame + num_frames
            features[sequence_idx, start_frame:end_frame] = time_warp_impl(
                features[sequence_idx, start_frame:end_frame], factor=time_warp_factor
            )

    return features
