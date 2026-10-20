"""Contact phase: labels from demonstrations and the gate that predicts them.

``rho(t) = exp(-d(t)^2 / 2 sigma^2)``, ``d`` = frames to the nearest gripper open/close switch in
the demonstration.  Labels come from the demo's own gripper command (no simulator state), the
same signal the DP experiments used for phase-aware gradient scaling.  The gate predicts the
largest ``rho`` over the action chunk about to be generated: "will this chunk touch something".
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def contact_phase(gripper: np.ndarray, sigma: float) -> np.ndarray:
    """Per-frame ``rho`` for one episode from its gripper command (sign = open / closed)."""
    state = np.asarray(gripper, dtype=np.float32) > 0
    events = np.flatnonzero(state[1:] != state[:-1]) + 1
    if len(events) == 0:
        return np.zeros(len(state), np.float32)
    frames = np.arange(len(state))[:, None]
    distance = np.abs(frames - events[None]).min(1).astype(np.float32)
    return np.exp(-(distance**2) / (2 * sigma**2)).astype(np.float32)


class PhaseGate(nn.Module):
    """Pooled structure tokens (+ proprio) -> contact-phase logit."""

    def __init__(self, dim: int, state_dim: Optional[int] = None, hidden: int = 256):
        super().__init__()
        self.state = nn.Linear(state_dim, dim) if state_dim else None
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, pooled: Tensor, proprio: Optional[Tensor] = None) -> Tensor:
        x = pooled
        if self.state is not None and proprio is not None:
            x = x + self.state(proprio.reshape(proprio.shape[0], -1).to(x.dtype))
        return self.head(x).squeeze(-1)
