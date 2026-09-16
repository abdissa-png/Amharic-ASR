"""Conformer + linear CTC head."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.conformer import ConformerEncoder, build_from_config, count_parameters


class ConformerCTC(nn.Module):
    def __init__(self, encoder: ConformerEncoder, vocab_size: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.ctc = nn.Linear(encoder.d_model, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, h_len, _ = self.encoder(feats, lengths)
        logits = self.ctc(h)  # (B, T, V)
        return logits, h_len

    def loss(
        self,
        feats: torch.Tensor,
        lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        *,
        inter_weight: float = 0.5,
        blank: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, h_len, h_mid = self.encoder(feats, lengths)
        logits = self.ctc(h)
        nll = self.ctc_loss(logits, h_len, targets, target_lengths, blank=blank)
        if h_mid is not None and inter_weight:
            nll = nll + float(inter_weight) * self.ctc_loss(self.ctc(h_mid), h_len, targets, target_lengths, blank=blank)
        return nll, logits, h_len

    def ctc_loss(
        self,
        logits: torch.Tensor,
        logit_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        blank: int = 0,
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, V)
        return F.ctc_loss(
            log_probs,
            targets,
            logit_lengths,
            target_lengths,
            blank=blank,
            reduction="mean",
            zero_infinity=True,
        )


def build_model(cfg: dict, vocab_size: int) -> ConformerCTC:
    encoder = build_from_config(cfg)
    return ConformerCTC(encoder, vocab_size)


def param_count(model: nn.Module) -> int:
    return count_parameters(model)
