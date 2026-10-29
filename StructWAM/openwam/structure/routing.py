"""Phase-gated routing of the action stream's attention.

The joint attention runs over ``[video | action | structure]``.  For action queries only, a
per-sample additive bias is put on the keys:

    structure keys    + w_structure * (1 - g)     far from contact: follow the geometry
    video keys        + w_video     * g           near contact: read the imagined future
    wrist-tile video  + w_wrist     * g           ... and the close-up camera in particular

``g`` in [0, 1] is the contact phase (predicted, or the demo label under teacher forcing).
The weights are learned (logit units; init 1 / 1 / 0.5).  Only the action rows get a float
mask; the other rows keep the boolean mask, so the driver runs them through the fused
(flash) attention and only the few action rows through the slower float-mask path.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class RoutingBias(nn.Module):
    def __init__(self, init_structure: float = 1.0, init_video: float = 1.0, init_wrist: float = 0.5):
        super().__init__()
        self.weights = nn.Parameter(torch.tensor([init_structure, init_video, init_wrist]))

    def forward(
        self,
        action_rows: Tensor,
        gate: Tensor,
        *,
        s_video: int,
        s_action: int,
        wrist_video: Optional[Tensor] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Boolean mask rows of the action queries ``[s_action, S]`` or ``[B, 1, s_action, S]``
        -> float mask ``[B, 1, s_action, S]`` with the routing bias added."""
        batch = gate.shape[0]
        if action_rows.ndim == 2:
            action_rows = action_rows.unsqueeze(0).unsqueeze(0)
        action_rows = action_rows.expand(batch, 1, *action_rows.shape[-2:])
        floor = torch.finfo(dtype).min / 2
        out = torch.zeros(action_rows.shape, dtype=dtype, device=action_rows.device).masked_fill(~action_rows, floor)
        g = gate.to(dtype=dtype).view(batch, 1, 1, 1)
        w_structure, w_video, w_wrist = self.weights.to(dtype)
        video_bias = w_video * g
        if wrist_video is not None:
            video_bias = video_bias + w_wrist * g * wrist_video.to(device=out.device, dtype=dtype).view(1, 1, 1, -1)
        structure_bias = (w_structure * (1 - g)).expand(batch, 1, s_action, out.shape[-1] - s_video - s_action)
        bias = torch.cat((video_bias.expand(batch, 1, s_action, s_video),
                          torch.zeros(batch, 1, s_action, s_action, dtype=dtype, device=out.device),
                          structure_bias), -1)
        return out + bias
