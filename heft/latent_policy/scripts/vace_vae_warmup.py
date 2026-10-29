"""Warm up the native Wan VAE without shifting retained observation endpoints."""

from __future__ import annotations

import numpy as np
import torch


def observation_window_indices(video_frames, start, length, warmup_frames=0):
    if warmup_frames < 0 or warmup_frames % 4:
        raise ValueError("VAE warm-up must be a nonnegative multiple of four frames")
    if length < 1 or (length - 1) % 4:
        raise ValueError("Retained RGB length must be 4n+1")
    if start < 0 or start + length > video_frames:
        raise ValueError("Requested observation window is outside the video")
    # Negative history indices repeat the earliest available frame, never future frames.
    indices = np.maximum(np.arange(start - warmup_frames, start + length), 0)
    metadata = {"warmup_frames": int(warmup_frames),
                "history_frames": int(min(start, warmup_frames)),
                "repeated_prefix_frames": int(max(0, warmup_frames - start)),
                "discarded_latent_positions": int(warmup_frames // 4),
                "frame_indices": np.arange(start, start + length, 4)}
    return indices, metadata


@torch.no_grad()
def encode_observation_pixels(vae, pixels, warmup_frames=0, zero_latents=None):
    """Encode [3,warmup+4n+1,H,W], returning only the retained latent positions."""
    if pixels.ndim != 4 or pixels.shape[0] != 3:
        raise ValueError("Expected pixels with shape [3,T,H,W]")
    full_length = pixels.shape[1]
    retained_length = full_length - warmup_frames
    if warmup_frames < 0 or warmup_frames % 4 or retained_length < 1 or (retained_length - 1) % 4:
        raise ValueError("Warm-up must be 4m frames and retained clip must be 4n+1 frames")
    observed = vae.encode([pixels])[0]
    cache = {} if zero_latents is None else zero_latents
    key = (full_length, pixels.shape[2], pixels.shape[3])
    if key not in cache:
        cache[key] = vae.encode([torch.zeros_like(pixels)])[0].detach().cpu()
    reactive = cache[key].to(pixels.device)
    offset = warmup_frames // 4
    observed, reactive = observed[:, offset:], reactive[:, offset:]
    expected_time = (retained_length - 1) // 4 + 1
    if observed.shape != reactive.shape or observed.shape[1] != expected_time:
        raise RuntimeError("VAE warm-up crop did not preserve the retained output length")
    return observed.contiguous(), reactive.contiguous()


@torch.no_grad()
def observation_control(vae, pixels, warmup_frames=0, zero_latents=None):
    observed, reactive = encode_observation_pixels(vae, pixels, warmup_frames, zero_latents)
    mask = torch.zeros((64, *observed.shape[1:]), device=observed.device, dtype=observed.dtype)
    return torch.cat((observed, reactive, mask), dim=0).unsqueeze(0).bfloat16()
