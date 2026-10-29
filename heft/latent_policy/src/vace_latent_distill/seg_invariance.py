"""Appearance-only consistency for partition K, with no external object labels."""

import torch
from torch.nn import functional as F

from .config import HEAD_TASKS, head_view
from .losses import masked_mean

MODES = ("none", "color", "texture", "alternating")


def appearance_augmentation(rgb, kind):
    if rgb.dtype != torch.uint8 or rgb.ndim != 5 or rgb.shape[2] != 3:
        raise ValueError("Expected uint8 video [B,T,3,H,W]")
    if kind not in ("color", "texture"):
        raise ValueError("Appearance augmentation must be color or texture")
    value = rgb.float()
    if kind == "color":
        shape = (len(rgb), 1, 1, 1, 1)
        brightness = 0.8 + 0.4 * torch.rand(shape, device=rgb.device)
        contrast = 0.8 + 0.4 * torch.rand(shape, device=rgb.device)
        saturation = 0.6 + 0.8 * torch.rand(shape, device=rgb.device)
        gray_mix = (torch.rand(shape, device=rgb.device) < 0.1).float()
        gray_mix = gray_mix * 0.35 * torch.rand(shape, device=rgb.device)
        # One transform per clip avoids introducing artificial temporal flicker.
        center = value.mean((1, 2, 3, 4), keepdim=True)
        value = (value - center) * contrast + center
        luminance = value[:, :, :1] * 0.299 + value[:, :, 1:2] * 0.587 + value[:, :, 2:] * 0.114
        value = luminance + (value - luminance) * saturation * (1 - gray_mix)
        value = value * brightness
    else:
        clips = []
        height, width = value.shape[-2:]
        axis = torch.arange(-1, 2, device=rgb.device, dtype=torch.float32)
        for clip in value:
            sigma = 0.4 + 0.4 * torch.rand((), device=rgb.device)
            kernel = torch.exp(-axis.square() / (2 * sigma.square()))
            kernel = kernel / kernel.sum()
            kernel = (kernel[:, None] * kernel[None, :]).expand(3, 1, 3, 3)
            softened = F.conv2d(F.pad(clip, (1, 1, 1, 1), mode="replicate"), kernel, groups=3)
            ratio = float(0.85 + 0.15 * torch.rand((), device=rgb.device))
            small = (max(1, round(height * ratio)), max(1, round(width * ratio)))
            softened = F.interpolate(softened, small, mode="bilinear", align_corners=False, antialias=True)
            softened = F.interpolate(softened, (height, width), mode="bilinear", align_corners=False)
            # Retain fine boundaries; there is no crop, warp or region mask.
            blend = 0.25 + 0.35 * torch.rand((), device=rgb.device)
            clips.append(clip * (1 - blend) + softened * blend)
        value = torch.stack(clips)
    return value.round().clamp(0, 255).to(torch.uint8)


def seg_invariance_loss(augmented, clean, task, head_weights=None, *, future_weight=1.0):
    if task != "shared" and not task.startswith("seg_"):
        raise ValueError("Partition invariance requires shared or seg training")
    if future_weight < 0:
        raise ValueError("Future consistency weight must be nonnegative")
    heads = tuple(name for name in HEAD_TASKS if name.startswith("seg_")) if task == "shared" else (task,)
    weights = dict.fromkeys(heads, 1.0) if head_weights is None or task != "shared" else head_weights
    # Keep all shared outputs connected for DDP/FSDP, with zero direct loss on
    # non-partition heads. Only the clean branch is used as a stopped target.
    total = sum(value.sum() * 0 for branch in ("observed", "future")
                for value in augmented[branch].values())
    details = {}
    for name in heads:
        terms = []
        for branch in ("observed", "future"):
            prediction = (head_view(augmented[branch], name) if task == "shared"
                          else augmented[branch])["seg_k"]
            reference = (head_view(clean[branch], name) if task == "shared"
                         else clean[branch])["seg_k"].detach()
            valid = augmented["timeline"][branch]["valid"][None]
            error = (1 - F.cosine_similarity(prediction.float(), reference.float(), dim=-1)).clamp_min(0)
            loss = masked_mean(error, valid)
            details[f"{name}/{branch}"] = loss.detach()
            terms.append(loss * (future_weight if branch == "future" else 1.0))
        total = total + weights[name] * sum(terms) / (1 + future_weight)
    details["total"] = total.detach()
    return total, details
