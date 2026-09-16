"""Compact Conformer encoder for CTC ASR.

Default 20M-class recipe: d_model=256, layers=12, heads=4, ffn=1024, conv=31.

Optional knobs (all off by default, so existing checkpoints still load):

- ``pos_encoding=rel`` — Transformer-XL relative attention (Gulati/ESPnet).
- ``subsample_factor=2`` — one stride-2 conv; the encoder stays on a 20 ms grid.
- ``subsample_factor=8`` + ``subsample=dw_striding`` — Fast Conformer front-end.
- ``conv_kernel_size=[9, 15, 31]`` — parallel depthwise kernels (Multi-Convformer).
- ``conv_norm=layer_norm`` — no batch statistics in the conv module.
- ``zipformer=true`` — U-Net temporal stacks (Zipformer-inspired, not icefall).
- ``att_context_size`` — local attention window in encoder frames; -1 = global.
  Past a few seconds it is computed in query blocks, never as a T×T matrix.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


KernelSpec = Union[int, Sequence[int]]


def _ceil_div(n: int, k: int) -> int:
    return math.ceil(n / k)


def _stride_lengths(lengths: torch.Tensor, n: int) -> torch.Tensor:
    out = lengths
    for _ in range(n):
        out = torch.div(out - 1, 2, rounding_mode="floor") + 1
    return out


def zip_schedule(num_layers: int) -> list[tuple[int, int]]:
    """Layer counts at relative rates 1×, 2×, 4×, 2×, 1× (vs the subsampled grid)."""
    if num_layers < 4:
        raise ValueError("zipformer needs at least 4 layers")
    if num_layers == 4:
        return [(1, 1), (1, 2), (1, 2), (1, 1)]
    a = max(1, num_layers // 6)
    c = num_layers - 4 * a
    if c < 1:
        a = 1
        c = num_layers - 4
    return [(a, 1), (a, 2), (c, 4), (a, 2), (a, 1)]


class Conv2dSubsampling(nn.Module):
    """Time–frequency strided conv front-end.

    The default (factor=4, style=conv2d) is byte-compatible with existing
    20 M checkpoints: two stride-2 Conv2d layers, then a linear projection.
    Factor 2 keeps both conv2d layers but the second strides frequency only,
    so the Conformer runs at 20 ms and every weight has the factor-4 shape
    (a 4× checkpoint loads in full). Fast Conformer uses factor=8 and
    depthwise-separable extra layers.
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int,
        factor: int = 4,
        style: str = "conv2d",
    ) -> None:
        super().__init__()
        if factor not in (2, 4, 8):
            raise ValueError("subsample_factor must be 2, 4, or 8")
        if style not in ("conv2d", "dw_striding"):
            raise ValueError("subsample must be conv2d or dw_striding")
        self.time_ratio = factor
        n_time = int(math.log2(factor))
        n_layers = max(2, n_time) if style == "conv2d" else n_time
        layers: list[nn.Module] = []
        for i in range(n_layers):
            if i == 0 or style == "conv2d":
                stride = 2 if i < n_time else (1, 2)
                layers.append(nn.Conv2d(1 if i == 0 else d_model, d_model, kernel_size=3, stride=stride, padding=1))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1, groups=d_model))
                layers.append(nn.Conv2d(d_model, d_model, kernel_size=1))
                layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)
        freq_out = in_dim
        for _ in range(n_layers):
            freq_out = _ceil_div(freq_out, 2)
        self.out = nn.Linear(d_model * freq_out, d_model)
        self._n_stride = n_time

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)
        cur = lengths
        for layer in self.conv:
            x = layer(x)
            if isinstance(layer, nn.Conv2d) and layer.stride[0] == 2:
                cur = _stride_lengths(cur, 1)
            elif isinstance(layer, nn.ReLU):
                # Bias makes padded frames nonzero; the next conv would read them.
                pad = torch.arange(x.size(2), device=x.device)[None, :] >= cur[:, None]
                x = x.masked_fill(pad[:, None, :, None], 0.0)
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * f)
        x = self.out(x)
        return x, cur


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0)


class RelPositionalEncoding(nn.Module):
    """Transformer-XL relative PE used by Conformer / ESPnet / NeMo ``rel_pos``."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.register_buffer("pe", self._build(max_len), persistent=False)

    def _build(self, max_len: int) -> torch.Tensor:
        ext = 2 * max_len - 1
        pe = torch.zeros(ext, self.d_model)
        pos = torch.arange(max_len - 1, -max_len, -1, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        if t > self.max_len:
            self.max_len = t
            pe = self._build(t).to(device=x.device, dtype=self.pe.dtype)
            self.register_buffer("pe", pe, persistent=False)
        center = self.pe.size(0) // 2
        return self.pe[center - t + 1 : center + t].unsqueeze(0)


def _rel_shift(x: torch.Tensor) -> torch.Tensor:
    """(B, H, T, 2T-1) relative logits → (B, H, T, T) aligned to keys."""
    b, h, t, pos = x.shape
    x = F.pad(x, (1, 0))
    x = x.view(b, h, pos + 1, t)
    x = x[:, :, 1:].view(b, h, t, pos)
    return x[:, :, :, :t]


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, att_context_size: int = -1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.att_context_size = int(att_context_size)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.norm(x)
        attn_mask = _local_mask(x.size(1), self.att_context_size, x.device, x.dtype)
        out, _ = self.attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        return self.drop(out)


class RelSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, att_context_size: int = -1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        self.att_context_size = int(att_context_size)
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)
        self.u = nn.Parameter(torch.zeros(n_heads, self.d_k))
        self.v = nn.Parameter(torch.zeros(n_heads, self.d_k))
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _shape(self, x: torch.Tensor, b: int, t: int) -> torch.Tensor:
        return x.view(b, t, self.n_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.norm(x)
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = self._shape(q, b, t), self._shape(k, b, t), self._shape(v, b, t)
        c = self.att_context_size
        if c >= 0 and t > _block_size(c) + 2 * c:
            merged = self._attend_blocked(q, k, v, pos_emb, key_padding_mask, c)
        else:
            merged = self._attend_full(q, k, v, pos_emb, key_padding_mask)
        return self.drop(self.out(merged))

    def _attend_full(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pos_emb: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        b, _, t, _ = q.shape
        ac = torch.matmul(q + self.u[None, :, None, :], k.transpose(-2, -1))
        if pos_emb is None:
            scores = ac * self.scale
        else:
            p = self._shape(self.pos_proj(pos_emb).expand(b, -1, -1), b, pos_emb.size(1))
            bd = torch.matmul(q + self.v[None, :, None, :], p.transpose(-2, -1))
            scores = (ac + _rel_shift(bd)) * self.scale
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        local = _local_mask(t, self.att_context_size, q.device, q.dtype)
        if local is not None:
            scores = scores + local.unsqueeze(0).unsqueeze(0)
        attn = self.drop(torch.softmax(scores, dim=-1))
        attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.matmul(attn, v)
        return out.transpose(1, 2).contiguous().view(b, t, -1)

    def _attend_blocked(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pos_emb: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        c: int,
    ) -> torch.Tensor:
        """Local attention as batched matmuls over query blocks.

        Queries are cut into blocks of ``n`` frames. Each block scores the
        ``n + 2c`` keys around it, and the band mask keeps ±c. That wastes
        some scores at block edges but stays on tensor-core matmuls; a
        per-query window (``unfold``) had a backward that dominated the step.
        """
        b, h, t, d = q.shape
        n = _block_size(c)
        nb = -(-t // n)
        span = n + 2 * c
        dev = q.device

        q_blk = F.pad(q, (0, 0, 0, nb * n - t)).view(b, h, nb, n, d)
        k_blk = _key_blocks(k, n, c)
        v_blk = _key_blocks(v, n, c)
        scores = torch.matmul(q_blk + self.u[None, :, None, None, :], k_blk.transpose(-2, -1))

        a = torch.arange(n, device=dev)[:, None]
        s = torch.arange(span, device=dev)[None, :]
        if pos_emb is not None:
            # Key slot s of query a sits at offset s - c - a; P rows cover
            # every offset a block can see, then gather picks each pair.
            reach = n + c - 1
            offsets = torch.arange(-reach, reach + 1, device=dev)
            rows = (t - 1 + offsets).clamp(0, pos_emb.size(1) - 1)
            p = self._shape(self.pos_proj(pos_emb.index_select(1, rows)), 1, offsets.numel())
            bd = torch.matmul(q_blk + self.v[None, :, None, None, :], p.unsqueeze(2).transpose(-2, -1))
            pick = (s - a + n - 1).expand(b, h, nb, n, span)
            scores = scores + torch.gather(bd, -1, pick)
        scores = scores * self.scale

        key_pos = torch.arange(nb, device=dev)[:, None, None] * n - c + s[None]
        blocked = ((s - c - a).abs() > c)[None] | (key_pos < 0) | (key_pos >= t)
        scores = scores.masked_fill(blocked[None, None], float("-inf"))
        if key_padding_mask is not None:
            pad = _key_blocks(key_padding_mask[:, None, :, None].to(q.dtype), n, c, value=1.0)
            scores = scores.masked_fill(pad[..., 0][:, :, :, None, :] > 0.5, float("-inf"))

        attn = self.drop(torch.softmax(scores, dim=-1))
        attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.matmul(attn, v_blk).view(b, h, nb * n, d)[:, :, :t]
        return out.transpose(1, 2).contiguous().view(b, t, -1)


def _block_size(c: int) -> int:
    return max(64, c)


def _key_blocks(x: torch.Tensor, n: int, c: int, value: float = 0.0) -> torch.Tensor:
    """(..., T, d) → (..., ceil(T/n), n + 2c, d): each query block's ±c keys."""
    t, d = x.size(-2), x.size(-1)
    nb = -(-t // n)
    x = F.pad(x, (0, 0, c, nb * n - t + c), value=value)
    lead = x.shape[:-2]
    left = x[..., : nb * n, :].reshape(*lead, nb, n, d)[..., :c, :]
    mid = x[..., c : c + nb * n, :].reshape(*lead, nb, n, d)
    right = x[..., 2 * c : 2 * c + nb * n, :].reshape(*lead, nb, n, d)[..., n - c :, :]
    return torch.cat([left, mid, right], dim=-2)


def _local_mask(t: int, context: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    if context is None or context < 0:
        return None
    idx = torch.arange(t, device=device)
    band = (idx[None, :] - idx[:, None]).abs() > context
    return band.to(dtype=dtype).masked_fill(band, torch.finfo(dtype).min)


class ConvModule(nn.Module):
    def __init__(
        self,
        d_model: int,
        kernel_size: KernelSpec,
        dropout: float,
        conv_norm: str = "batch_norm",
    ) -> None:
        super().__init__()
        kernels = (kernel_size,) if isinstance(kernel_size, int) else tuple(int(k) for k in kernel_size)
        if not kernels or any(k < 1 or k % 2 == 0 for k in kernels):
            raise ValueError("conv kernels must be positive odd integers")
        if conv_norm not in ("batch_norm", "layer_norm"):
            raise ValueError("conv_norm must be batch_norm or layer_norm")
        self.kernels = kernels
        self.conv_norm = conv_norm
        self.norm = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.dw = nn.ModuleList(
            [
                nn.Conv1d(d_model, d_model, k, padding=(k - 1) // 2, groups=d_model)
                for k in kernels
            ]
        )
        if len(kernels) == 1:
            # Keep the historical name so 20 M checkpoints load.
            self.dw = self.dw[0]
        if conv_norm == "batch_norm":
            self.bn = nn.BatchNorm1d(d_model)
        else:
            self.bn = nn.LayerNorm(d_model)
        self.act = nn.SiLU()
        self.pw2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def _depthwise(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.dw, nn.ModuleList):
            return sum(conv(x) for conv in self.dw) / len(self.dw)
        return self.dw(x)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv_norm == "batch_norm":
            return self.bn(x)
        return self.bn(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm(x).transpose(1, 2)
        x = self.glu(self.pw1(x))
        if pad_mask is not None:
            # Padded frames carry nonzero activations after the first layer;
            # zeroing them makes a clip's output independent of its batch.
            x = x.masked_fill(pad_mask[:, None, :], 0.0)
        x = self._depthwise(x)
        x = self.act(self._norm(x))
        x = self.drop(self.pw2(x))
        return x.transpose(1, 2)


class TimeDownsample(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return x, _stride_lengths(lengths, 1)


class TimeUpsample(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        skip_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.interpolate(x.transpose(1, 2), size=skip.size(1), mode="nearest").transpose(1, 2)
        return x + skip, skip_lengths


class ConformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        conv_kernel: KernelSpec,
        dropout: float,
        *,
        pos_encoding: str = "abs",
        conv_norm: str = "batch_norm",
        att_context_size: int = -1,
    ) -> None:
        super().__init__()
        self.ff1 = FeedForward(d_model, ffn_dim, dropout)
        if pos_encoding == "rel":
            self.attn = RelSelfAttention(d_model, n_heads, dropout, att_context_size)
        else:
            self.attn = SelfAttention(d_model, n_heads, dropout, att_context_size)
        self.conv = ConvModule(d_model, conv_kernel, dropout, conv_norm=conv_norm)
        self.ff2 = FeedForward(d_model, ffn_dim, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(x, key_padding_mask, pos_emb=pos_emb)
        x = x + self.conv(x, key_padding_mask)
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 80,
        d_model: int = 256,
        num_layers: int = 8,
        num_heads: int = 4,
        ffn_dim: int = 1024,
        conv_kernel_size: KernelSpec = 31,
        dropout: float = 0.1,
        *,
        subsample_factor: int = 4,
        subsample: str = "conv2d",
        pos_encoding: str = "abs",
        conv_norm: str = "batch_norm",
        zipformer: bool = False,
        att_context_size: int = -1,
        inter_layer: int | None = 6,
    ) -> None:
        super().__init__()
        if pos_encoding not in ("abs", "rel"):
            raise ValueError("pos_encoding must be abs or rel")
        self.subsample = Conv2dSubsampling(input_dim, d_model, factor=subsample_factor, style=subsample)
        self.pos_encoding = pos_encoding
        self.zipformer = bool(zipformer)
        self.inter_layer = inter_layer
        self.d_model = d_model
        self.time_ratio = self.subsample.time_ratio
        if pos_encoding == "rel":
            self.pos = RelPositionalEncoding(d_model)
        else:
            self.pos = SinusoidalPE(d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    d_model,
                    num_heads,
                    ffn_dim,
                    conv_kernel_size,
                    dropout,
                    pos_encoding=pos_encoding,
                    conv_norm=conv_norm,
                    att_context_size=att_context_size,
                )
                for _ in range(num_layers)
            ]
        )
        self.schedule = zip_schedule(num_layers) if self.zipformer else [(num_layers, 1)]
        self.down = nn.ModuleDict()
        self.up = nn.ModuleDict()
        if self.zipformer:
            seen = {1}
            for _, rate in self.schedule:
                if rate not in seen:
                    parent = rate // 2
                    self.down[f"{parent}_{rate}"] = TimeDownsample(d_model)
                    self.up[f"{rate}_{parent}"] = TimeUpsample()
                    seen.add(rate)

    def _mask(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        return torch.arange(t, device=x.device)[None, :] >= lengths[:, None]

    def _maybe_pos(self, x: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.pos_encoding == "rel":
            return self.dropout(x), self.pos(x)
        return self.dropout(self.pos(x)), None

    def forward(
        self,
        feats: torch.Tensor,
        lengths: torch.Tensor,
        inter_layer: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if inter_layer is None:
            inter_layer = self.inter_layer
        x, lengths = self.subsample(feats, lengths)
        x, pos_emb = self._maybe_pos(x)
        h_mid = None
        if not self.zipformer:
            mask = self._mask(x, lengths)
            for i, layer in enumerate(self.layers, start=1):
                x = layer(x, mask, pos_emb=pos_emb)
                if inter_layer is not None and i == inter_layer:
                    h_mid = x
            return x, lengths, h_mid

        skips: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        rate = 1
        layer_i = 0
        for n_layers, target in self.schedule:
            while rate < target:
                skips[rate] = (x, lengths)
                x, lengths = self.down[f"{rate}_{rate * 2}"](x, lengths)
                rate *= 2
                pos_emb = self.pos(x) if self.pos_encoding == "rel" else None
            while rate > target:
                skip, skip_len = skips[rate // 2]
                x, lengths = self.up[f"{rate}_{rate // 2}"](x, skip, skip_len)
                rate //= 2
                pos_emb = self.pos(x) if self.pos_encoding == "rel" else None
            mask = self._mask(x, lengths)
            for _ in range(n_layers):
                x = self.layers[layer_i](x, mask, pos_emb=pos_emb)
                layer_i += 1
                if inter_layer is not None and layer_i == inter_layer:
                    h_mid = x
        if h_mid is not None and h_mid.size(1) != x.size(1):
            h_mid = F.interpolate(h_mid.transpose(1, 2), size=x.size(1), mode="nearest").transpose(1, 2)
        return x, lengths, h_mid


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _kernel_spec(raw) -> KernelSpec:
    if isinstance(raw, (list, tuple)):
        return [int(k) for k in raw]
    return int(raw)


def build_from_config(cfg: dict) -> ConformerEncoder:
    m = cfg["model"]
    return ConformerEncoder(
        input_dim=int(m.get("input_dim", 80)),
        d_model=int(m["d_model"]),
        num_layers=int(m["num_layers"]),
        num_heads=int(m["num_heads"]),
        ffn_dim=int(m["ffn_dim"]),
        conv_kernel_size=_kernel_spec(m.get("conv_kernel_size", 31)),
        dropout=float(m.get("dropout", 0.1)),
        subsample_factor=int(m.get("subsample_factor", 4)),
        subsample=str(m.get("subsample", "conv2d")),
        pos_encoding=str(m.get("pos_encoding", "abs")),
        conv_norm=str(m.get("conv_norm", "batch_norm")),
        zipformer=bool(m.get("zipformer", False)),
        att_context_size=int(m.get("att_context_size", -1)),
        inter_layer=int(m["inter_layer"]) if m.get("inter_layer") is not None else 6,
    )


if __name__ == "__main__":
    import argparse
    from utils import load_yaml

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_yaml(args.config)
    enc = build_from_config(cfg)
    head = nn.Linear(enc.d_model, int(cfg.get("vocab_size", 400)))
    n = count_parameters(enc) + count_parameters(head)
    print(f"encoder+ctc_head ≈ {n:,} parameters ({n/1e6:.2f}M)")
    print(
        f"subsample={enc.time_ratio}x pos={enc.pos_encoding} zip={enc.zipformer} "
        f"layers={len(enc.layers)}"
    )
