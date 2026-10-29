"""Structure features -> structure-expert tokens.

Per camera: 64 map tokens (student maps, 4x4 patches of the 32x32 grid), 32 keypoint tokens
(filtered tracks, 13 steps), 16 trunk tokens (V-JEPA2 trunk pooled to 4x4, with a learned
attention pool over all 17 time steps), and one token per selected instruction word (its UMT5
embedding plus its attention concentration) — the word's map is among the map channels, so the
expert can tie "bowl" / "plate" to where each points without any grammar.  A missing camera or
an unselected word keeps its slots, masked out, so the sequence length is fixed.  ``groups``
selects which groups are emitted (all four by default); a left-out group has no tokens, and
its parameters stay in place so checkpoints load either way.
"""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from openwam.structure.layout import TOKEN_DIM

MAP_PATCH = 4
TRUNK_DIM = 1024
TRUNK_POOL = 4
KEYPOINTS = 32
TEXT_DIM = 4096
GROUPS = ("map", "keypoint", "trunk", "word")


class StructureTokenizer(nn.Module):
    def __init__(self, dim: int, cameras: Sequence[str] = ("head", "wrist"), map_channels: int = 37,
                 word_slots: int = 5, groups: Sequence[str] = GROUPS):
        super().__init__()
        self.cameras = tuple(cameras)
        unknown = set(groups) - set(GROUPS)
        if unknown or not groups:
            raise ValueError(f"structure token groups must be a non-empty subset of {GROUPS}, got {list(groups)}")
        self.groups = tuple(g for g in GROUPS if g in groups)
        self.map_norm = nn.GroupNorm(map_channels, map_channels)
        self.map_embed = nn.Conv2d(map_channels, dim, MAP_PATCH, MAP_PATCH)
        self.keypoint_embed = nn.Sequential(nn.LayerNorm(TOKEN_DIM), nn.Linear(TOKEN_DIM, dim), nn.SiLU(),
                                            nn.Linear(dim, dim))
        self.trunk_time = nn.Linear(TRUNK_DIM, 1)
        self.trunk_embed = nn.Sequential(nn.LayerNorm(TRUNK_DIM), nn.Linear(TRUNK_DIM, dim))
        self.word_embed = nn.Sequential(nn.LayerNorm(TEXT_DIM), nn.Linear(TEXT_DIM, dim))
        self.word_score = nn.Linear(1, dim)
        grid = 32 // MAP_PATCH
        self.map_pos = nn.Parameter(torch.zeros(grid * grid, dim))
        self.trunk_pos = nn.Parameter(torch.zeros(TRUNK_POOL * TRUNK_POOL, dim))
        self.word_pos = nn.Parameter(torch.zeros(max(word_slots, 1), dim))
        self.group_embed = nn.Parameter(torch.zeros(len(GROUPS), dim))
        self.camera_embed = nn.Parameter(torch.zeros(len(self.cameras), dim))
        for parameter in (self.map_pos, self.trunk_pos, self.word_pos, self.group_embed, self.camera_embed):
            nn.init.normal_(parameter, std=0.02)
        self.counts = {"map": grid * grid, "keypoint": KEYPOINTS, "trunk": TRUNK_POOL * TRUNK_POOL,
                       "word": int(word_slots)}
        self.counts = {g: (n if g in self.groups else 0) for g, n in self.counts.items()}

    @property
    def tokens_per_camera(self) -> int:
        return sum(self.counts.values())

    def slices(self) -> dict:
        """``{(camera, group): slice}`` into the token sequence."""
        out, start = {}, 0
        for camera in self.cameras:
            for group in GROUPS:
                out[camera, group] = slice(start, start + self.counts[group])
                start += self.counts[group]
        return out

    def _trunk(self, trunk: Tensor) -> Tensor:
        """``[B, 17, 16, 16, 1024]`` -> ``[B, 16, 1024]``: 4x4 spatial pool, attention pool over time."""
        batch, steps = trunk.shape[:2]
        pooled = F.adaptive_avg_pool2d(trunk.float().flatten(0, 1).permute(0, 3, 1, 2), TRUNK_POOL)
        pooled = pooled.flatten(2).transpose(1, 2).unflatten(0, (batch, steps))        # [B, 17, 16, 1024]
        weights = self.trunk_time(pooled.to(self.trunk_time.weight.dtype)).softmax(1)
        return (weights * pooled.to(weights.dtype)).sum(1)

    def _group(self, group: str, value: dict, batch: int, device) -> tuple:
        count = self.counts[group]
        dtype = self.map_pos.dtype
        if group == "map":
            maps = self.map_norm(value["maps"].to(device=device, dtype=self.map_embed.weight.dtype))
            tokens = self.map_embed(maps).flatten(2).transpose(1, 2) + self.map_pos
            return tokens, torch.ones(batch, count, dtype=torch.bool, device=device)
        if group == "keypoint":
            tokens = self.keypoint_embed(value["keypoints"].to(device=device, dtype=dtype))
            return tokens, value["keypoint_use"].to(device=device, dtype=torch.bool)
        if group == "trunk":
            tokens = self.trunk_embed(self._trunk(value["trunk"].to(device))) + self.trunk_pos
            return tokens, torch.ones(batch, count, dtype=torch.bool, device=device)
        if "word_embedding" not in value:
            return torch.zeros(batch, count, self.map_pos.shape[1], device=device, dtype=dtype), \
                torch.zeros(batch, count, dtype=torch.bool, device=device)
        tokens = (self.word_embed(value["word_embedding"].to(device=device, dtype=dtype))
                  + self.word_score(value["word_score"].to(device=device, dtype=dtype).unsqueeze(-1))
                  + self.word_pos[:count])
        return tokens, value["word_mask"].to(device=device, dtype=torch.bool)

    def forward(self, features: dict, batch: int, dtype=None) -> dict:
        """``{camera: student features}`` -> tokens ``[B, S, dim]``, mask ``[B, S]``, pooled ``[B, dim]``."""
        dtype = dtype or self.map_pos.dtype
        device = self.map_pos.device
        tokens, masks = [], []
        for index, camera in enumerate(self.cameras):
            value = features.get(camera)
            for group_index, group in enumerate(GROUPS):
                count = self.counts[group]
                if count == 0:
                    continue
                if value is None:
                    tokens.append(torch.zeros(batch, count, self.map_pos.shape[1], device=device, dtype=dtype))
                    masks.append(torch.zeros(batch, count, dtype=torch.bool, device=device))
                    continue
                group_tokens, mask = self._group(group, value, batch, device)
                group_tokens = group_tokens + self.group_embed[group_index] + self.camera_embed[index]
                tokens.append(group_tokens.to(dtype))
                masks.append(mask)
        tokens, mask = torch.cat(tokens, 1), torch.cat(masks, 1)
        # Keep at least one valid token per sample (a fully masked row would NaN in SDPA).
        mask[:, 0] = mask[:, 0] | ~mask.any(1)
        weights = mask.to(tokens.dtype).unsqueeze(-1)
        pooled = (tokens * weights).sum(1) / weights.sum(1).clamp(min=1)
        return {"tokens": tokens, "mask": mask, "pooled": pooled}
