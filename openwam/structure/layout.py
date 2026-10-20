"""Shared geometry: the student's keypoint-token layout and the multiview canvas.

Keypoint tokens (``latent_dp.policy_maps.keypoint_tokens``) are ``[P, 57]``:
13 steps x (dx, dy, similarity, valid) with (dx, dy) = (xy - present) / 4 in 32-grid cells,
then the present position (cell + 0.5) / 32 * 2 - 1, then the role one-hot (hand, segments of the
selected words now, at +16).  Step 8 is the present; steps 9..12 are +4, +8, +12, +16 frames ahead.

The OpenWAM LIBERO canvas (``multiview=true``, layout ``[head, left wrist, None]``) puts the
head camera in the top two thirds over the full width and the wrist camera in the bottom-left
third-height / half-width tile (384x320: 256x320 and 128x160).  Every camera frame is square
(256x256) and is stretched into its tile.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

GRID = 32
STEPS = 13
PRESENT = 8
STEP_FRAMES = (-32, -28, -24, -20, -16, -12, -8, -4, 0, 4, 8, 12, 16)
ROLES = ("hand", "words_now", "words_+16")      # word mode; the role mode's (hand, target, destination) has the same size
TOKEN_DIM = STEPS * 4 + 2 + len(ROLES)


@dataclass(frozen=True)
class KeypointLayout:
    """Decode keypoint tokens back into cell coordinates."""

    @staticmethod
    def split(tokens: Tensor) -> dict:
        """``tokens [..., P, 57]`` -> xy cells ``[..., P, 13, 2]``, similarity / valid ``[..., P, 13]``,
        role ``[..., P]`` (-1 = empty slot)."""
        steps = tokens[..., : STEPS * 4].unflatten(-1, (STEPS, 4))
        present = ((tokens[..., STEPS * 4 : STEPS * 4 + 2] + 1) / 2) * GRID - 0.5
        xy = present.unsqueeze(-2) + 4.0 * steps[..., :2]
        role_hot = tokens[..., STEPS * 4 + 2 :]
        role = torch.where(role_hot.sum(-1) > 0, role_hot.argmax(-1), torch.full_like(role_hot[..., 0], -1).long())
        return {"xy": xy, "similarity": steps[..., 2], "valid": steps[..., 3] > 0.5, "role": role}

    @staticmethod
    def step_index(frames_ahead: int) -> int:
        return STEP_FRAMES.index(int(frames_ahead))


@dataclass(frozen=True)
class CanvasLayout:
    """Camera tiles of the multiview canvas, as fractions of its height / width."""

    tiles: tuple = (("head", 0.0, 2 / 3, 0.0, 1.0), ("wrist", 2 / 3, 1.0, 0.0, 0.5))

    def tile(self, camera: str):
        for name, top, bottom, left, right in self.tiles:
            if name == camera:
                return top, bottom, left, right
        raise KeyError(f"camera {camera!r} has no canvas tile")

    def cells_to_canvas(self, xy_cells: Tensor, camera: str) -> Tensor:
        """32-grid cell coordinates of a camera frame -> canvas coordinates in [0, 1]."""
        top, bottom, left, right = self.tile(camera)
        unit = (xy_cells + 0.5) / GRID
        x = left + unit[..., 0] * (right - left)
        y = top + unit[..., 1] * (bottom - top)
        return torch.stack((x, y), -1)

    def maps_to_grid(self, maps: dict, height: int, width: int) -> Tensor:
        """``{camera: [B, C, h, w]}`` -> ``[B, C, height, width]`` on any canvas-aligned grid
        (latent or DiT tokens); outside the tiles is zero."""
        reference = next(iter(maps.values()))
        out = reference.new_zeros(reference.shape[0], reference.shape[1], height, width)
        for camera, value in maps.items():
            top, bottom, left, right = self.tile(camera)
            r0, r1 = round(top * height), round(bottom * height)
            c0, c1 = round(left * width), round(right * width)
            if r1 > r0 and c1 > c0:
                out[:, :, r0:r1, c0:c1] = F.interpolate(value.float(), size=(r1 - r0, c1 - c0), mode="bilinear",
                                                        align_corners=False).to(out.dtype)
        return out

    def region_mask(self, camera: str, height: int, width: int, device=None) -> Tensor:
        """Bool ``[height, width]`` of the grid cells inside a camera tile."""
        top, bottom, left, right = self.tile(camera)
        mask = torch.zeros(height, width, dtype=torch.bool, device=device)
        mask[round(top * height) : round(bottom * height), round(left * width) : round(right * width)] = True
        return mask

    def crop(self, grid: Tensor, camera: str) -> Tensor:
        """``[..., H, W, D]`` token grid -> the camera tile's sub-grid."""
        height, width = grid.shape[-3], grid.shape[-2]
        top, bottom, left, right = self.tile(camera)
        return grid[..., round(top * height) : round(bottom * height), round(left * width) : round(right * width), :]
