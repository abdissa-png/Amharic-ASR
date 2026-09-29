"""CTC knowledge distillation from a larger teacher (Phase 14).

Two modes:

* ``token`` — KL(teacher ‖ student) over aligned frame logits. Requires the
  same vocabulary. Used when the teacher is a fine-tuned copy of the same
  output units.
* ``sequence`` — treat teacher transcripts as extra supervised CTC targets
  (works across vocabularies).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def token_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    lengths: torch.Tensor,
    *,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Frame-level KD. Logits are (B, T, V). Teacher is detached."""
    t = temperature
    s = F.log_softmax(student_logits / t, dim=-1)
    q = F.softmax(teacher_logits.detach() / t, dim=-1)
    kl = F.kl_div(s, q, reduction="none").sum(-1)  # (B, T)
    b, tmax = kl.shape
    mask = torch.arange(tmax, device=kl.device)[None, :] < lengths[:, None]
    return (kl * mask).sum() / mask.sum().clamp_min(1) * (t * t)


def mixed_loss(ctc: torch.Tensor, kd: torch.Tensor, supervised_weight: float) -> torch.Tensor:
    """``supervised_weight`` is the GT CTC share: 1.0, 0.75, or 0.5."""
    w = float(supervised_weight)
    return w * ctc + (1.0 - w) * kd
