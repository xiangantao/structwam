"""Structure-only action sketch and schedule shaping (no pixels are ever generated).

``SketchHead`` predicts the whole action chunk from structure alone (pooled structure tokens,
the hand keypoints' tracks of both cameras, proprio).  It is trained with the same normalized
actions as the action stream, and serves three purposes:

* warm start: the action denoiser starts from the sketch noised to ``sigma0`` instead of from
  pure noise, so the geometry sets the gross motion and the joint model refines it;
* difficulty: its per-sample error marks what structure alone cannot do (``losses``);
* ablation: "structure only" policy = the sketch.

``shape_schedule`` turns the engine's joint schedule into the StructWAM one: the video stream
takes only its first ``video_steps`` denoising steps and then stays put (its tokens still ride
along in the joint attention), and the action stream holds at ``action_start_t`` until the
schedule passes it.
"""

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from openwam.structure.layout import KeypointLayout, STEPS

HAND = 0


def hand_track_features(keypoints: Tensor, use: Tensor) -> Tensor:
    """``[B, P, 57]`` keypoint tokens -> ``[B, 13 * 4]`` mean step features of the hand keypoints."""
    parts = KeypointLayout.split(keypoints)
    hand = (parts["role"] == HAND) & use.bool()
    steps = keypoints[..., : STEPS * 4]
    weights = hand.to(steps.dtype).unsqueeze(-1)
    return (steps * weights).sum(1) / weights.sum(1).clamp(min=1)


class SketchHead(nn.Module):
    def __init__(self, dim: int, horizon: int, action_dim: int, cameras: Sequence[str],
                 state_dim: Optional[int] = None, hidden: int = 1024):
        super().__init__()
        self.horizon, self.action_dim, self.cameras = int(horizon), int(action_dim), tuple(cameras)
        inputs = dim + STEPS * 4 * len(self.cameras) + (int(state_dim) if state_dim else 0)
        self.state_dim = state_dim
        self.net = nn.Sequential(nn.LayerNorm(inputs), nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                                 nn.SiLU(), nn.Linear(hidden, self.horizon * self.action_dim))

    def forward(self, pooled: Tensor, features: dict, proprio: Optional[Tensor] = None) -> Tensor:
        parts = [pooled]
        for camera in self.cameras:
            value = features.get(camera)
            if value is None:
                parts.append(pooled.new_zeros(pooled.shape[0], STEPS * 4))
            else:
                parts.append(hand_track_features(value["keypoints"].to(pooled.dtype), value["keypoint_use"]))
        if self.state_dim:
            state = proprio.reshape(pooled.shape[0], -1).to(pooled.dtype) if proprio is not None else \
                pooled.new_zeros(pooled.shape[0], self.state_dim)
            parts.append(state)
        x = torch.cat(parts, -1).to(self.net[1].weight.dtype)
        return self.net(x).view(-1, self.horizon, self.action_dim)


def masked_action_mse(prediction: Tensor, target: Tensor, action_is_pad: Optional[Tensor]) -> Tensor:
    """Per-sample MSE ``[B]`` with OpenWAM's (B, T) or (B, T, D) padding masks."""
    error = (prediction.float() - target.float()) ** 2
    if action_is_pad is None:
        return error.mean(dim=(1, 2))
    valid = (~action_is_pad.to(device=error.device, dtype=torch.bool)).float()
    if valid.shape != error.shape:
        valid = (valid > 0).any(-1).float() if valid.ndim == 3 else valid
        valid = valid.unsqueeze(-1).expand_as(error)
    return (error * valid).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp(min=1)


def warm_start(sketch: Tensor, scheduler, t_start: float, generator: Optional[torch.Generator] = None) -> Tensor:
    """Noise the sketch to the action sigma of ``t_start`` (flow matching: (1 - s) x + s eps)."""
    sigma = float(t_start) / float(scheduler.num_train_timesteps)
    noise = torch.randn(sketch.shape, device=sketch.device, dtype=sketch.dtype, generator=generator)
    return scheduler.add_noise(sketch, noise, torch.tensor(sigma, device=sketch.device, dtype=sketch.dtype))


def shape_schedule(schedule: Sequence[Tuple[float, float]], *, video_steps: Optional[int] = None,
                   action_start_t: Optional[float] = None) -> List[Tuple[float, float]]:
    """Limit the video stream to its first ``video_steps`` updates; hold actions at ``action_start_t``."""
    entries = [(float(t_v), float(t_a)) for t_v, t_a in schedule]
    out = []
    for index, (t_v, t_a) in enumerate(entries):
        if video_steps is not None:
            t_v = entries[min(index, int(video_steps))][0]
        if action_start_t is not None:
            t_a = min(t_a, float(action_start_t))
        out.append((t_v, t_a))
    return out
