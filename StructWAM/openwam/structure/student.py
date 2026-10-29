"""Frozen VACE-distilled student -> per-camera structure features.

Wraps ``latent_dp.online.OnlineMaps`` from the latent_policy project (not vendored: the student
and its post-processing keep a single source of truth there).  The student is a plain attribute
of the architecture, never an ``nn.Module`` child, so it is not trained, frozen-listed, or written
into OpenWAM checkpoints; deploy hosts need the same paths.

StructWAM runs the student in word mode (no grammar, no per-task files): the instruction's
content words are ranked by attention concentration per window and up to five are kept; any
instruction works (missing ones are UMT5-encoded on the fly).  Per camera and window:

* ``maps``          ``[B, 37, 32, 32]`` (``WORD_MAP_CHANNELS``): word heat now / +16 for up to five
  words, filtered K-K flow at five horizons, motion, reliability, camera motion, hand / word
  segments now / +16 / kept masks, appearance PCA, depth,
* ``keypoints``     ``[B, 32, 57]`` filtered keypoint tracks (see ``layout.KeypointLayout``),
* ``keypoint_use``  ``[B, 32]`` bool,
* ``trunk``         ``[B, 17, 16, 16, 1024]`` the fine-tuned V-JEPA2 trunk (all 17 time steps),
* ``word_embedding`` ``[B, 5, 4096]`` UMT5 embedding of each selected word (mean of its tokens),
  ``word_score`` ``[B, 5]`` its concentration, ``word_mask`` ``[B, 5]``.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

CAMERA_IS_WRIST = {"head": False, "wrist": True}


@dataclass
class StudentConfig:
    latent_policy_root: str              # .../heft/latent_policy (src/ holds latent_dp and vace_latent_distill)
    checkpoint: str                      # student checkpoint, e.g. runs/mainline_20261016/g2_full/step_0002000.pt
    appearance_stats: str                # npz with k_mean / k_basis (present-K PCA), e.g. a DP run's stats.npz
    roles: Optional[str] = None          # legacy grammar roles (role mode); None = word mode
    text_root: Optional[str] = None      # UMT5 tokenizer / encoder; None = the student checkpoint's teacher text_root
    cameras: Sequence[str] = ("head", "wrist")
    history: int = 33
    image_size: int = 256
    workers: int = 8


def _add_paths(config: StudentConfig) -> None:
    root = Path(config.latent_policy_root)
    for path in (root / "src", root.parent):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def map_channel_names(config: StudentConfig) -> tuple:
    """The student's map channels for this configuration (word or role mode)."""
    _add_paths(config)
    from latent_dp.policy_maps import MAP_CHANNELS, WORD_MAP_CHANNELS

    return MAP_CHANNELS if config.roles else WORD_MAP_CHANNELS


def word_slots(config: StudentConfig) -> int:
    _add_paths(config)
    from latent_dp.policy_maps import WORD_SLOTS

    return 0 if config.roles else WORD_SLOTS


class StructureStudent:
    def __init__(self, config: StudentConfig, device):
        self.config = config
        self.device = torch.device(device)
        self._online = None

    @property
    def online(self):
        if self._online is None:
            _add_paths(self.config)
            from latent_dp.online import OnlineMaps

            online = OnlineMaps(self.config.checkpoint, self.config.roles, self.device, self.config.workers,
                                text_root=self.config.text_root)
            with np.load(self.config.appearance_stats) as stats:
                online.k_basis = (stats["k_mean"].astype(np.float32), stats["k_basis"].astype(np.float32))
            self._online = online
        return self._online

    @property
    def map_channels(self) -> tuple:
        return map_channel_names(self.config)

    def map_channel(self, name: str) -> int:
        return self.map_channels.index(name)

    @torch.no_grad()
    def encode(self, frames: dict, prompts: Sequence[str]) -> dict:
        """``{camera: uint8 [B, T, H, W, 3] or [B, T, 3, H, W]}`` + prompts -> ``{camera: features}``."""
        cameras = [c for c in self.config.cameras if frames.get(c) is not None]
        if not cameras:
            raise ValueError("structure student needs at least one camera history")
        clips, tasks, wrist = [], [], []
        for camera in cameras:
            clip = torch.as_tensor(frames[camera])
            if clip.shape[-1] == 3:
                clip = clip.permute(0, 1, 4, 2, 3)
            if clip.shape[1] != self.config.history or clip.dtype != torch.uint8:
                raise ValueError(f"{camera}: expected uint8 [B, {self.config.history}, 3, H, W], got "
                                 f"{tuple(clip.shape)} {clip.dtype}")
            clips.append(clip)
            tasks += list(prompts)
            wrist += [CAMERA_IS_WRIST[camera]] * len(prompts)
        rgb = torch.cat(clips).to(self.device, non_blocking=True)
        result = self.online(rgb, tasks, wrist)
        maps, tokens, use, trunk = result[:4]
        words = result[5] if len(result) > 5 else None
        batch = len(prompts)
        out = {}
        for index, camera in enumerate(cameras):
            part = slice(index * batch, (index + 1) * batch)
            value = {
                "maps": torch.from_numpy(np.ascontiguousarray(maps[part])).to(self.device, torch.float32),
                "keypoints": torch.from_numpy(np.ascontiguousarray(tokens[part])).to(self.device, torch.float32),
                "keypoint_use": torch.from_numpy(np.ascontiguousarray(use[part])).to(self.device),
                "trunk": trunk[part],
            }
            if words is not None:
                value["word_embedding"] = words["embedding"][part].float()
                value["word_score"] = torch.from_numpy(words["score"][part]).to(self.device)
                value["word_mask"] = torch.from_numpy(words["mask"][part]).to(self.device)
            out[camera] = value
        return out

    def close(self) -> None:
        if self._online is not None:
            self._online.close()
            self._online = None
