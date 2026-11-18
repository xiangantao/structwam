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

CAMERA_IS_WRIST = {"head": False, "wrist": True, "wrist_right": True, "head_right": False}   # wrist_right: bimanual (RoboTwin)


@dataclass
class StudentConfig:
    latent_policy_root: str              # .../heft/latent_policy (src/ holds latent_dp and vace_latent_distill)
    checkpoint: str                      # student checkpoint, e.g. $STRUCTWAM_ASSETS/student/step_0002000.pt
    appearance_stats: str                # npz with k_mean / k_basis (present-K PCA), e.g. a DP run's stats.npz
    roles: Optional[str] = None          # legacy grammar roles (role mode); None = word mode
    text_root: Optional[str] = None      # UMT5 tokenizer / encoder; None = the student checkpoint's teacher text_root
    cameras: Sequence[str] = ("head", "wrist")
    history: int = 33
    image_size: int = 256
    workers: int = 8
    async_maps: bool = False             # training: overlap the CPU map pool with the video / text encoding
    maps_profile: Optional[str] = None   # policy_maps.HAND_THRESHOLDS key, e.g. "robocasa"; None = LIBERO
    chunk: Optional[int] = None          # max windows per student forward (peak device memory); None = all cameras at once
    # "vace": the distilled structure student (maps, keypoints, trunk, words).  "vjepa_raw": representation
    # ablation — the official V-JEPA2 ViT-L (vjepa_weights) on the same 33 frames, trunk only, no text fusion.
    encoder: str = "vace"
    vjepa_root: Optional[str] = None     # vjepa_raw: V-JEPA2 source checkout
    vjepa_weights: Optional[str] = None  # vjepa_raw: official vitl.pt (target_encoder)


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
                                text_root=self.config.text_root, maps_profile=self.config.maps_profile)
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
        return self.collect(self.launch(frames, prompts))

    @torch.no_grad()
    def launch(self, frames: dict, prompts: Sequence[str]):
        """First half of ``encode``: student forward on the device; the CPU map pool keeps running."""
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
        chunk = int(self.config.chunk or 0)
        if chunk <= 0 or rgb.shape[0] <= chunk:
            pending = self.online(rgb, tasks, wrist, defer=True)
        else:   # e.g. RoboTwin's 3 cameras x batch 8 = 24 windows do not fit next to the training state at once
            pending = [self.online(rgb[i : i + chunk], tasks[i : i + chunk], wrist[i : i + chunk], defer=True)
                       for i in range(0, rgb.shape[0], chunk)]
        return {"pending": pending, "cameras": cameras, "batch": len(prompts)}

    def _collect_pending(self, pending):
        """``OnlineMaps.collect`` of one launch, or of its chunks concatenated in window order."""
        if not isinstance(pending, list):
            return self.online.collect(pending)
        parts = [self.online.collect(p) for p in pending]

        def cat(values):
            return torch.cat(values) if torch.is_tensor(values[0]) else np.concatenate(values)

        out = tuple(cat([part[i] for part in parts]) for i in range(5))
        if len(parts[0]) > 5:
            words = {key: cat([part[5][key] for part in parts]) for key in ("embedding", "score", "mask")}
            words["text"] = [w for part in parts for w in part[5]["text"]]
            out = out + (words,)
        return out

    @torch.no_grad()
    def collect(self, launched) -> dict:
        """Second half of ``encode``."""
        result = self._collect_pending(launched["pending"])
        cameras, batch_size = launched["cameras"], launched["batch"]
        maps, tokens, use, trunk = result[:4]
        words = result[5] if len(result) > 5 else None
        batch = batch_size
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


class RawVJEPA:
    """Representation ablation: the official V-JEPA2 ViT-L in place of the distilled student.

    Same input as the student's trunk (``StructuredStudent.encode_past``): the 33 history frames at
    256, the last frame repeated to 34 (tubelet 2), ImageNet normalization, patch embed, all 24 blocks,
    final norm -> ``trunk [B, 17, 16, 16, 1024]`` (bf16).  No text fusion (the student fuses the
    instruction into its last 4 blocks) and no maps / keypoints / words: pair it with
    ``structure_expert.groups=[trunk]``, ``losses.video_saliency=0`` and ``losses.difficulty[1]=0``.
    Frozen and outside the module tree, as the student.
    """

    def __init__(self, config: StudentConfig, device):
        self.config = config
        self.device = torch.device(device)
        self._backbone = None

    @property
    def backbone(self):
        if self._backbone is None:
            _add_paths(self.config)
            from types import SimpleNamespace

            from vace_latent_distill.model import load_backbone

            if not (self.config.vjepa_root and self.config.vjepa_weights):
                raise ValueError("student.encoder=vjepa_raw needs student.vjepa_root and student.vjepa_weights")
            spec = SimpleNamespace(vjepa_root=self.config.vjepa_root, vjepa_weights=self.config.vjepa_weights,
                                   image_size=int(self.config.image_size), dim=1024)
            backbone = load_backbone(spec, pretrained=True).requires_grad_(False).eval()
            self._backbone = backbone.to(self.device)
            self._mean = torch.tensor((0.485, 0.456, 0.406), device=self.device).reshape(1, 3, 1, 1, 1)
            self._std = torch.tensor((0.229, 0.224, 0.225), device=self.device).reshape(1, 3, 1, 1, 1)
        return self._backbone

    @property
    def map_channels(self) -> tuple:
        return ()

    @torch.no_grad()
    def _trunk(self, rgb: torch.Tensor) -> torch.Tensor:
        """uint8 ``[N, 33, 3, S, S]`` on device -> ``[N, 17, S/16, S/16, 1024]`` bf16."""
        backbone = self.backbone
        pixels = torch.cat((rgb, rgb[:, -1:]), 1).permute(0, 2, 1, 3, 4).float() / 255
        pixels = (pixels - self._mean) / self._std
        temporal, height, width = pixels.shape[2] // 2, pixels.shape[3] // 16, pixels.shape[4] // 16
        with torch.autocast("cuda", dtype=torch.bfloat16):
            value = backbone.patch_embed(pixels)
            for block in backbone.blocks:
                value = block(value, mask=None, attn_mask=None, T=temporal, H_patches=height, W_patches=width)
            value = backbone.norm(value)
        return value.reshape(len(pixels), temporal, height, width, -1).to(torch.bfloat16)

    @torch.no_grad()
    def launch(self, frames: dict, prompts: Sequence[str]):
        cameras = [c for c in self.config.cameras if frames.get(c) is not None]
        if not cameras:
            raise ValueError("structure encoder needs at least one camera history")
        clips = []
        for camera in cameras:
            clip = torch.as_tensor(frames[camera])
            if clip.shape[-1] == 3:
                clip = clip.permute(0, 1, 4, 2, 3)
            if clip.shape[1] != self.config.history or clip.dtype != torch.uint8:
                raise ValueError(f"{camera}: expected uint8 [B, {self.config.history}, 3, H, W], got "
                                 f"{tuple(clip.shape)} {clip.dtype}")
            clips.append(clip)
        rgb = torch.cat(clips).to(self.device, non_blocking=True)
        chunk = int(self.config.chunk or 0) or rgb.shape[0]
        trunk = torch.cat([self._trunk(rgb[i : i + chunk]) for i in range(0, rgb.shape[0], chunk)])
        return {"trunk": trunk, "cameras": cameras, "batch": len(prompts)}

    @torch.no_grad()
    def collect(self, launched) -> dict:
        batch = launched["batch"]
        return {camera: {"trunk": launched["trunk"][index * batch : (index + 1) * batch]}
                for index, camera in enumerate(launched["cameras"])}

    @torch.no_grad()
    def encode(self, frames: dict, prompts: Sequence[str]) -> dict:
        return self.collect(self.launch(frames, prompts))

    def close(self) -> None:
        self._backbone = None
