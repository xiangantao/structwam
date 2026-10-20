"""How the structure stream shapes the video stream's training (all in latent space).

* ``video_weights``: spend the video loss where the structure says it matters — a per-pixel
  weight from the student's hand, motion and selected-word maps (the words the instruction's
  attention concentrates on; no grammar), placed on the canvas latent grid, times a per-sample difficulty weight (near contact, unsure tracks, high
  structure-only sketch error).  Both are normalized to mean 1, so the loss scale is unchanged.
* ``track_consistency``: the imagined next latent must agree with the student's predicted
  point tracks — features at a keypoint's present position in the clean conditioning latent
  should reappear where the track says it will be in the predicted (x0) next latent.
* ``alignment_loss``: REPA-style — a projection of the video DiT's hidden state on the current
  frame should match the student's V-JEPA2 trunk on the same camera tile.
"""

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from openwam.structure.layout import CanvasLayout, KeypointLayout

SALIENCY_CHANNELS = ("hand", "motion") + tuple(f"word_{i}_now" for i in range(1, 6))


def _unit(maps: Tensor) -> Tensor:
    """Per-sample, per-channel min-max to [0, 1]: ``[B, C, H, W]``."""
    low = maps.amin(dim=(2, 3), keepdim=True)
    high = maps.amax(dim=(2, 3), keepdim=True)
    return (maps - low) / (high - low).clamp(min=1e-6)


def saliency(maps: Tensor, channels: Sequence[int]) -> Tensor:
    """Student maps ``[B, 42, 32, 32]`` -> ``[B, 1, 32, 32]`` in [0, 1]."""
    return _unit(maps[:, list(channels)].float()).amax(1, keepdim=True)


def video_weights(features: dict, channels: Sequence[int], layout: CanvasLayout, latent_hw, strength: float) -> Tensor:
    """Per-pixel video-loss weight ``[B, 1, 1, H, W]`` on the latent grid, mean 1 per sample."""
    maps = {camera: saliency(value["maps"], channels) for camera, value in features.items()}
    grid = layout.maps_to_grid(maps, *latent_hw)
    weight = 1.0 + float(strength) * grid
    weight = weight / weight.mean(dim=(1, 2, 3), keepdim=True)
    return weight.unsqueeze(2)


def difficulty(phase: Tensor, similarity: Tensor, sketch_error: Optional[Tensor], coefficients) -> Tensor:
    """Per-sample video-loss weight ``[B]``, mean 1: near contact, unsure tracks, structure-only error."""
    near, unsure, residual = (float(c) for c in coefficients)
    weight = 1.0 + near * phase.float() + unsure * (1.0 - similarity.float().clamp(0, 1))
    if sketch_error is not None and residual > 0:
        error = sketch_error.detach().float()
        weight = weight + residual * error / error.mean().clamp(min=1e-6)
    return weight / weight.mean().clamp(min=1e-6)


def mean_track_similarity(features: dict) -> Tensor:
    """Average future-step similarity of the used keypoints, over cameras: ``[B]``."""
    values = []
    for value in features.values():
        parts = KeypointLayout.split(value["keypoints"])
        future = parts["similarity"][..., KeypointLayout.step_index(4):]
        use = value["keypoint_use"].bool().unsqueeze(-1).expand_as(future).float()
        values.append((future * use).sum(dim=(1, 2)) / use.sum(dim=(1, 2)).clamp(min=1))
    return torch.stack(values).mean(0)


def weighted_video_mse(noise_pred: Tensor, target: Tensor, timestep_weight: Tensor, *, n_skip: int,
                       video_is_pad: Optional[Tensor], pixel_weight: Optional[Tensor],
                       sample_weight: Optional[Tensor]) -> Tensor:
    """``BaseWAMArchitecture._compute_video_loss`` with optional pixel and sample weights."""
    if n_skip > 0:
        noise_pred, target = noise_pred[:, :, n_skip:], target[:, :, n_skip:]
    per_element = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
    if pixel_weight is not None:
        per_element = per_element * pixel_weight.to(per_element.device, per_element.dtype)
    per_frame = per_element.mean(dim=(1, 3, 4))
    if video_is_pad is not None:
        valid = ~video_is_pad.to(device=per_frame.device, dtype=torch.bool)
        per_sample = (per_frame * valid.float()).sum(1) / valid.float().sum(1).clamp(min=1)
    else:
        per_sample = per_frame.mean(1)
    if sample_weight is not None:
        per_sample = per_sample * sample_weight.to(per_sample.device, per_sample.dtype)
    return (per_sample * timestep_weight).mean()


def _sample(latent: Tensor, points: Tensor) -> Tensor:
    """``latent [B, C, H, W]``, canvas points in [0, 1] ``[B, P, 2]`` -> ``[B, P, C]``."""
    grid = (points * 2 - 1).unsqueeze(2).to(latent.dtype)
    return F.grid_sample(latent, grid, mode="bilinear", align_corners=False).squeeze(-1).transpose(1, 2)


def track_consistency(x0_pred: Tensor, x0_clean: Tensor, features: dict, layout: CanvasLayout,
                      future_steps: Sequence[int] = (4, 8, 12, 16), min_similarity: float = 0.6) -> Tensor:
    """Appearance carried along predicted tracks, from latent 0 (clean) to latent 1 (predicted).

    With video stride 4 and the 4x temporal VAE, latent 1 covers raw frames +4..+16 — the
    student's future horizon; its position is the mean of the track over those steps.
    """
    losses, weights = [], []
    for camera, value in features.items():
        parts = KeypointLayout.split(value["keypoints"].float())
        steps = [KeypointLayout.step_index(f) for f in future_steps]
        present = layout.cells_to_canvas(parts["xy"][..., KeypointLayout.step_index(0), :], camera)
        future = layout.cells_to_canvas(parts["xy"][..., steps, :].mean(-2), camera)
        ok = (value["keypoint_use"].bool() & parts["valid"][..., steps].all(-1)
              & (parts["similarity"][..., steps].amin(-1) >= min_similarity)).float()
        now = _sample(x0_clean[:, :, 0].float(), present).detach()
        ahead = _sample(x0_pred[:, :, 1].float(), future)
        cosine = F.cosine_similarity(now, ahead, dim=-1)
        losses.append(((1 - cosine) * ok).sum())
        weights.append(ok.sum())
    return torch.stack(losses).sum() / torch.stack(weights).sum().clamp(min=1)


def hidden_grid(hidden: Tensor, frames: int, height: int, width: int) -> Tensor:
    """Video DiT hidden state -> ``[B, T, H, W, D]`` (Cosmos is already 5-D; Wan is ``[B, L, D]``)."""
    if hidden.ndim == 5:
        return hidden
    return hidden.reshape(hidden.shape[0], frames, height, width, hidden.shape[-1])


def alignment_loss(hidden: Tensor, projector, features: dict, layout: CanvasLayout) -> Tensor:
    """REPA on the current frame: ``1 - cos(proj(h_tile), trunk_present)`` per camera tile."""
    losses = []
    for camera, value in features.items():
        tile = layout.crop(hidden[:, 0], camera)                                    # [B, h, w, D]
        projected = projector(tile.to(next(projector.parameters()).dtype))         # [B, h, w, 1024]
        target = value["trunk"][:, -1].float()                                      # [B, 16, 16, 1024]
        projected = F.interpolate(projected.permute(0, 3, 1, 2).float(), size=target.shape[1:3], mode="bilinear",
                                  align_corners=False).permute(0, 2, 3, 1)
        losses.append((1 - F.cosine_similarity(projected, target.detach(), dim=-1)).mean())
    return torch.stack(losses).mean()
