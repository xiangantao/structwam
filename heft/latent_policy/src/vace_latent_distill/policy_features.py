"""Post-processed, policy-facing features from the structured student (no raw Q/K).

For one camera and one decision time (the last of the student's 33 observed frames,
which is the student's present slot: future slot 8):

* K blocks: the present L22H9 | L15H7 K field is clustered (k-means, then connected
  components); large regions (table, wall) are background.  Up to ``BLOCKS`` blocks are
  kept, ranked by their relevance to the task's selected instruction tokens.
* Per block: centre, area, box, mean K (compressed to a few PCA dims later), wrist depth,
  relevance to each selected token (log of mean spatially-normalized word heat in the block),
  and the centre trajectory over the past 8 and future 4 native steps.
* Point tracks: up to ``POINTS`` points (each block's centre cell plus extra points in the
  most relevant blocks), tracked HEFT-style (K-K on the track K, x4 upsampling, local
  search, sub-cell argmax, EMA descriptor) forward through the future branch and backward
  through the observed branch.  Positions are displacements from the present, in units of
  the 256-px frame; the third channel is the descriptor similarity (confidence).
"""

import numpy as np
import torch
from torch.nn import functional as F

GRID = 32
BLOCKS = 8
POINTS = 32
WORDS = 3                        # selected instruction words per task (fewer are zero-padded)
PRESENT = 8                      # future slot of the last observed frame
OBSERVED = tuple(range(4, 12))   # observed native times with real frames (window frames 0..28)
FUTURE = tuple(range(8, 13))     # future slots 8..12 (window frames 32..48)
TRAJECTORY = (0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12)  # sequence indices kept: 8 past + 4 future
BACKGROUND_AREA = {False: 0.30, True: 0.60}   # third-person / wrist (a held object can be large)
BORDER_PENALTY = 1.0                           # third-person: big border-touching blocks (wall, table) rank last
MIN_CELLS = 3


def kmeans(values, clusters=12, iterations=10, seed=0):
    """Deterministic k-means (k-means++ init) on [N, C] float tensors -> labels [N]."""
    generator = torch.Generator().manual_seed(seed)
    first = int(torch.randint(len(values), (1,), generator=generator))
    centres = [values[first]]
    distance = ((values - centres[0]) ** 2).sum(-1)
    for _ in range(clusters - 1):
        index = int(torch.multinomial(distance / distance.sum().clamp_min(1e-12), 1, generator=generator))
        centres.append(values[index])
        distance = torch.minimum(distance, ((values - values[index]) ** 2).sum(-1))
    centres = torch.stack(centres)
    for _ in range(iterations):
        labels = torch.cdist(values, centres).argmin(1)
        for k in range(clusters):
            members = labels == k
            if members.any():
                centres[k] = values[members].mean(0)
    return torch.cdist(values, centres).argmin(1)


def k_blocks(k_field, word_heat, depth, wrist):
    """``k_field`` [cells, 256] unit halves, ``word_heat`` [cells, W], ``depth`` [cells] (numpy/torch CPU).

    Returns a list of dicts (ranked) with member cells and block descriptors.
    """
    from scipy import ndimage

    k_field = torch.as_tensor(k_field, dtype=torch.float32)
    labels = kmeans(k_field).numpy().reshape(GRID, GRID)
    heat = np.asarray(word_heat, np.float32)
    heat = heat / np.maximum(heat.mean(0, keepdims=True), 1e-8)          # spatially normalized
    depth = np.asarray(depth, np.float32)
    depth = (depth - depth.mean()) / max(float(depth.std()), 1e-6)
    ys, xs = np.mgrid[0:GRID, 0:GRID]
    blocks = []
    for k in np.unique(labels):
        components, count = ndimage.label(labels == k)
        for c in range(1, count + 1):
            mask = components == c
            cells = np.flatnonzero(mask.ravel())
            area = len(cells) / (GRID * GRID)
            if len(cells) < MIN_CELLS or area > BACKGROUND_AREA[bool(wrist)]:
                continue
            x, y = xs.ravel()[cells], ys.ravel()[cells]
            relevance = np.log(heat[cells].mean(0) + 1e-6)
            border = x.min() == 0 or y.min() == 0 or x.max() == GRID - 1 or y.max() == GRID - 1
            penalty = BORDER_PENALTY if (not wrist and border and area > 0.03) else 0.0
            blocks.append({
                "cells": cells,
                "centre": np.array([(x.mean() + 0.5) / GRID, (y.mean() + 0.5) / GRID], np.float32),
                "size": np.array([area, (x.max() - x.min() + 1) / GRID, (y.max() - y.min() + 1) / GRID], np.float32),
                "k": k_field[cells].mean(0).numpy(),
                "depth": np.array([depth[cells].mean() if wrist else 0.0, float(wrist)], np.float32),
                "relevance": relevance.astype(np.float32),
                "score": float(relevance.max()) + 0.1 * area - penalty,
            })
    blocks.sort(key=lambda b: -b["score"])
    return blocks[:BLOCKS]


def choose_points(blocks, extra_blocks=4, per_block=6):
    """Each block's centre cell, then farthest-point extras in the most relevant blocks."""
    points, owner = [], []
    for slot, block in enumerate(blocks):
        cells = block["cells"]
        xy = np.stack((cells % GRID, cells // GRID), -1) + 0.5
        centre = block["centre"] * GRID
        points.append(cells[np.argmin(((xy - centre) ** 2).sum(-1))])
        owner.append(slot)
    for slot, block in enumerate(blocks[:extra_blocks]):
        cells = block["cells"]
        xy = np.stack((cells % GRID, cells // GRID), -1).astype(np.float32)
        chosen = [int(np.flatnonzero(cells == points[slot])[0])]
        for _ in range(min(per_block, len(cells) - 1)):
            if len(points) >= POINTS:
                break
            gap = ((xy[:, None] - xy[chosen][None]) ** 2).sum(-1).min(1)
            pick = int(gap.argmax())
            if gap[pick] <= 0:
                break
            chosen.append(pick)
            points.append(cells[pick])
            owner.append(slot)
    return np.asarray(points[:POINTS]), np.asarray(owner[:POINTS])


def track(sequence, start_xy, order, scale=4, search_px=32.0, argmax_px=8.0, ema=0.05, temperature=0.02):
    """K-K tracks through ``sequence[order]`` ([S, cells, C] device tensor), start = order[0].

    Returns pixel positions [P, len(order), 2] and descriptor similarity [P, len(order)].
    """
    device = sequence.device
    side = GRID * scale
    pixel = 256 / side
    frames = sequence[list(order)].float().reshape(len(order), GRID, GRID, -1).permute(0, 3, 1, 2)
    frames = F.normalize(F.interpolate(frames, scale_factor=scale, mode="bilinear", align_corners=False), dim=1)
    ys, xs = torch.meshgrid(torch.arange(side, device=device), torch.arange(side, device=device), indexing="ij")
    grid = torch.stack((xs, ys), -1).reshape(-1, 2).float()

    def sample(frame, xy):
        norm = ((xy + 0.5) / side * 2 - 1)[None, :, None]
        return F.normalize(F.grid_sample(frame[None], norm, align_corners=False)[0, :, :, 0].T, dim=1)

    position = torch.as_tensor(start_xy, dtype=torch.float32, device=device) / pixel
    descriptor = sample(frames[0], position)
    path, similarity = [position], [torch.ones(len(position), device=device)]
    for step in range(1, len(order)):
        flat = frames[step].reshape(frames.shape[1], -1)
        score = descriptor @ flat
        distance = torch.cdist(position, grid) * pixel
        score = score.masked_fill(distance > search_px, -2)
        peak = grid[score.argmax(1)]
        near = torch.cdist(peak, grid) * pixel <= argmax_px
        weights = ((score - score.max(1, keepdim=True).values) / temperature).exp() * near
        position = (weights @ grid) / weights.sum(1, keepdim=True)
        feature = sample(frames[step], position)
        similarity.append((descriptor * feature).sum(-1))
        descriptor = F.normalize((1 - ema) * descriptor + ema * feature, dim=1)
        path.append(position)
    return torch.stack(path, 1) * pixel, torch.stack(similarity, 1)


def block_inputs(future, tokens):
    """Device -> CPU inputs of ``k_blocks`` for one window (present K, word heat, depth)."""
    k_field = torch.cat((F.normalize(future["seg_k_l22h9"][PRESENT].float(), dim=-1),
                         F.normalize(future["seg_k_l15h7"][PRESENT].float(), dim=-1)), -1) / np.sqrt(2)
    attention = future["text_attention"][PRESENT].float()
    heat = torch.stack([attention[:, list(word)].sum(1) for word in tokens], 1)
    return k_field.cpu(), heat.cpu().numpy(), future["depth"][PRESENT].float().cpu().numpy()


def blocks_job(job):
    """Process-pool entry: ``(k_field, heat, depth, wrist)`` -> ranked blocks."""
    torch.set_num_threads(1)
    return k_blocks(*job)


def window_features(observed, future, tokens, wrist, blocks=None):
    """One window's features.  ``observed``/``future``: dicts of per-window student fields on device
    (``seg_k_l22h9``/``seg_k_l15h7``/``track_k`` [T, cells, C], ``text_attention`` [T, cells, L],
    ``depth`` [T, cells]); ``tokens``: the task's selected words, each a list of token indices."""
    if blocks is None:
        blocks = k_blocks(*block_inputs(future, tokens), wrist)
    result = {
        "valid": np.zeros(BLOCKS, bool), "centre": np.zeros((BLOCKS, 2), np.float32),
        "size": np.zeros((BLOCKS, 3), np.float32), "k": np.zeros((BLOCKS, 256), np.float16),
        "depth": np.zeros((BLOCKS, 2), np.float32), "relevance": np.zeros((BLOCKS, WORDS), np.float32),
        "trajectory": np.zeros((BLOCKS, len(TRAJECTORY), 2), np.float32), "confidence": np.zeros(BLOCKS, np.float32),
        "tracks": np.zeros((POINTS, len(TRAJECTORY), 3), np.float16), "owner": np.full(POINTS, -1, np.int8),
    }
    if not blocks:
        return result
    for slot, block in enumerate(blocks):
        result["valid"][slot] = True
        for key in ("centre", "size", "depth"):
            result[key][slot] = block[key]
        result["relevance"][slot, : len(block["relevance"])] = block["relevance"]
        result["k"][slot] = block["k"]
    cells, owner = choose_points(blocks)
    start = np.stack((cells % GRID, cells // GRID), -1) * 8 + 3.5
    # Sequence: observed t=4..11 (window frames 0..28), then future slots 8..12 (frames 32..48).
    sequence = torch.cat((observed["track_k"][list(OBSERVED)], future["track_k"][list(FUTURE)]), 0)
    forward, forward_sim = track(sequence, start, (8, 9, 10, 11, 12))
    backward, backward_sim = track(sequence, start, (8, 7, 6, 5, 4, 3, 2, 1, 0))
    positions = torch.cat((backward.flip(1)[:, :8], forward[:, 1:]), 1)             # indices TRAJECTORY
    similarity = torch.cat((backward_sim.flip(1)[:, :8], forward_sim[:, 1:]), 1)
    displacement = ((positions - torch.as_tensor(start, dtype=torch.float32, device=positions.device)[:, None]) / 256)
    displacement, similarity = displacement.cpu().numpy(), similarity.cpu().numpy()
    count = len(cells)
    result["tracks"][:count, :, :2] = displacement
    result["tracks"][:count, :, 2] = similarity
    result["owner"][:count] = owner
    for slot in range(len(blocks)):
        mine = owner == slot
        result["trajectory"][slot] = displacement[mine].mean(0)
        result["confidence"][slot] = similarity[mine].mean()
    return result

def track_many(sequences, starts, order, scale=4, search_px=32.0, argmax_px=8.0, ema=0.05, temperature=0.02):
    """Batched ``track``: ``sequences`` [W, S, cells, C] (device), ``starts`` [W, P, 2] pixels.

    Same arithmetic as ``track`` per window, all windows at once.  Returns pixel positions
    [W, P, len(order), 2] and descriptor similarity [W, P, len(order)].
    """
    device = sequences.device
    windows = sequences.shape[0]
    side = GRID * scale
    pixel = 256 / side
    frames = sequences[:, list(order)].float()                                          # [W, S', cells, C]
    channels = frames.shape[-1]
    frames = frames.reshape(windows * len(order), GRID, GRID, channels).permute(0, 3, 1, 2)
    frames = F.normalize(F.interpolate(frames, scale_factor=scale, mode="bilinear", align_corners=False), dim=1)
    frames = frames.reshape(windows, len(order), channels, side, side)
    ys, xs = torch.meshgrid(torch.arange(side, device=device), torch.arange(side, device=device), indexing="ij")
    grid = torch.stack((xs, ys), -1).reshape(-1, 2).float()                            # [N, 2]

    def sample(frame, xy):                                                             # frame [W, C, s, s], xy [W, P, 2]
        norm = ((xy + 0.5) / side * 2 - 1)[:, :, None]
        return F.normalize(F.grid_sample(frame, norm, align_corners=False)[..., 0].transpose(1, 2), dim=-1)

    position = torch.as_tensor(starts, dtype=torch.float32, device=device) / pixel        # [W, P, 2]
    descriptor = sample(frames[:, 0], position)
    path, similarity = [position], [torch.ones(position.shape[:2], device=device)]
    for step in range(1, len(order)):
        score = torch.einsum("wpc,wcn->wpn", descriptor, frames[:, step].reshape(windows, channels, -1))
        distance = torch.cdist(position, grid[None].expand(windows, -1, -1)) * pixel
        score = score.masked_fill(distance > search_px, -2)
        peak = grid[score.argmax(-1)]                                                  # [W, P, 2]
        near = torch.cdist(peak, grid[None].expand(windows, -1, -1)) * pixel <= argmax_px
        weights = ((score - score.max(-1, keepdim=True).values) / temperature).exp() * near
        position = (weights @ grid) / weights.sum(-1, keepdim=True)
        feature = sample(frames[:, step], position)
        similarity.append((descriptor * feature).sum(-1))
        descriptor = F.normalize((1 - ema) * descriptor + ema * feature, dim=-1)
        path.append(position)
    return torch.stack(path, 2) * pixel, torch.stack(similarity, 2)


def batch_features(observed_track_k, future_track_k, blocks_list, words_count=WORDS):
    """Features for W windows whose K blocks are already computed (``k_blocks`` per window).

    ``observed_track_k`` / ``future_track_k``: [W, T, cells, C] device tensors.  Same outputs
    as ``window_features`` per window, with the point tracks computed in one batched pass.
    """
    count = len(blocks_list)
    results, starts, owners, counts = [], np.zeros((count, POINTS, 2), np.float32), [], []
    for w, blocks in enumerate(blocks_list):
        result = {
            "valid": np.zeros(BLOCKS, bool), "centre": np.zeros((BLOCKS, 2), np.float32),
            "size": np.zeros((BLOCKS, 3), np.float32), "k": np.zeros((BLOCKS, 256), np.float16),
            "depth": np.zeros((BLOCKS, 2), np.float32), "relevance": np.zeros((BLOCKS, words_count), np.float32),
            "trajectory": np.zeros((BLOCKS, len(TRAJECTORY), 2), np.float32), "confidence": np.zeros(BLOCKS, np.float32),
            "tracks": np.zeros((POINTS, len(TRAJECTORY), 3), np.float16), "owner": np.full(POINTS, -1, np.int8),
        }
        for slot, block in enumerate(blocks):
            result["valid"][slot] = True
            for key in ("centre", "size", "depth"):
                result[key][slot] = block[key]
            result["relevance"][slot, : len(block["relevance"])] = block["relevance"]
            result["k"][slot] = block["k"]
        cells, owner = choose_points(blocks) if blocks else (np.zeros(0, int), np.zeros(0, int))
        if len(cells):
            points = np.stack((cells % GRID, cells // GRID), -1) * 8 + 3.5
            starts[w, : len(cells)] = points
            starts[w, len(cells):] = points[0]                                         # padding, ignored
        results.append(result)
        owners.append(owner)
        counts.append(len(cells))
    sequence = torch.cat((observed_track_k[:, list(OBSERVED)], future_track_k[:, list(FUTURE)]), 1)
    forward, forward_sim = track_many(sequence, starts, (8, 9, 10, 11, 12))
    backward, backward_sim = track_many(sequence, starts, (8, 7, 6, 5, 4, 3, 2, 1, 0))
    positions = torch.cat((backward.flip(2)[:, :, :8], forward[:, :, 1:]), 2)
    similarity = torch.cat((backward_sim.flip(2)[:, :, :8], forward_sim[:, :, 1:]), 2)
    displacement = ((positions - torch.as_tensor(starts, device=positions.device)[:, :, None]) / 256).cpu().numpy()
    similarity = similarity.cpu().numpy()
    for w, (result, owner, n) in enumerate(zip(results, owners, counts)):
        if not n:
            continue
        result["tracks"][:n, :, :2] = displacement[w, :n]
        result["tracks"][:n, :, 2] = similarity[w, :n]
        result["owner"][:n] = owner
        for slot in range(int(result["valid"].sum())):
            mine = owner == slot
            result["trajectory"][slot] = displacement[w, :n][mine].mean(0)
            result["confidence"][slot] = similarity[w, :n][mine].mean()
    return results
