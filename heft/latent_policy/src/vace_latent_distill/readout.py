"""Existing Q-to-K beam decoder, reusable for independent student matrices."""

import importlib
import sys
from pathlib import Path

import torch

from .config import TeacherConfig


def decode_tracks(query, key, seeds, anchors, *, region_scores=None, config=None):
    """One camera/sample. Seeds are grid-cell ids from the caller's prompt/K readout.

    Returns native-grid trajectories and masks, not interpolated RGB positions.
    There is no object-count assumption, GT mask, or optical flow.
    """
    config = config or TeacherConfig()
    script_root = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(script_root))
    demo = importlib.import_module("demo_main_prompt_k_tracking")
    temporal, cells, channels = query.shape
    if key.shape != query.shape or seeds.shape[0] != len(anchors):
        raise ValueError("Q/K/seeds shapes disagree")
    grid = round(cells**0.5)
    if grid * grid != cells:
        raise ValueError("Tracking field is not a square grid")
    q, k = query.detach().float().cpu().numpy(), key.detach().float().cpu().numpy()
    region = (torch.ones(temporal, cells) if region_scores is None else region_scores).cpu().numpy()
    result = torch.zeros(len(anchors), seeds.shape[1], temporal, 2)
    valid = torch.zeros(result.shape[:-1], dtype=torch.bool)
    for index, anchor in enumerate(anchors.tolist()):
        chosen = seeds[index].detach().cpu().numpy()
        if len(chosen) == 0 or chosen.min() < 0 or chosen.max() >= cells:
            raise ValueError("Invalid anchor seed cells")
        end = min(anchor + 3, temporal)
        path = demo.track_dense(
            q[:end].reshape(-1, channels),
            k[:end].reshape(-1, channels),
            chosen,
            anchor,
            grid,
            region[:end],
            topk=config.beam_topk,
            beam_size=config.beam_size,
            max_step=config.beam_max_step,
            motion_weight=config.motion_weight,
            local_weight=config.local_weight,
            region_weight=config.region_weight,
            acceleration_weight=config.acceleration_weight,
            descriptor_ema=config.descriptor_ema,
            direction="q_to_k",
        )
        path = demo.regularize_paths(
            path,
            max_step=config.regularize_max_step,
            max_acceleration=config.regularize_max_acceleration,
            passes=2,
        )
        result[index, :, anchor:end] = torch.from_numpy(path)
        valid[index, :, anchor:end] = True
    return result, valid


def trajectory_heatmaps(paths, valid, grid=32, sigma=0.65):
    axis = torch.arange(grid, device=paths.device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    xy = torch.stack((xx, yy), -1)
    distance = (paths[..., None, None, :] - xy).square().sum(-1)
    heat = torch.exp(-distance / (2 * sigma**2))
    heat = heat / heat.sum((-1, -2), keepdim=True).clamp_min(1e-7)
    return heat * valid[..., None, None]
