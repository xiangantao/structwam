# Image encoder parts copied from LeRobot 0.6.2 (lerobot/policies/diffusion/modeling_diffusion.py),
# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab, and The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
"""Vanilla Diffusion Policy observation encoder (the control for the latent policy).

Chi et al. 2023 image configuration: per camera a ResNet-18 trained from scratch with every
BatchNorm replaced by GroupNorm (groups = channels // 16), spatial softmax with 32 keypoints,
linear + ReLU to 64 features; random crop for training and center crop for evaluation.
Optionally (``rgb_imagenet``) the LeRobot 0.6 default: ImageNet-pretrained ResNet-18 with BatchNorm.
The conditioning is DP's: for each observation step ``[state, camera features...]``,
flattened over steps.  UNet, scheduler, loss and sampling are shared with
``LatentDiffusionPolicy``.
"""

import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F

from .model import DP_DEFAULTS, LatentDiffusionPolicy
from .unet import DiffusionConditionalUnet1d


class SpatialSoftmax(nn.Module):
    """Spatial soft-argmax keypoints (Finn et al.; robomimic / LeRobot port)."""

    def __init__(self, input_shape, num_kp=None):
        super().__init__()
        self._in_c, self._in_h, self._in_w = input_shape
        if num_kp is not None:
            self.nets = nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features):
        if self.nets is not None:
            features = self.nets(features)
        features = features.reshape(-1, self._in_h * self._in_w)
        attention = F.softmax(features, dim=-1)
        return (attention @ self.pos_grid).view(-1, self._out_c, 2)


def _replace_batchnorm(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(num_groups=child.num_features // 16, num_channels=child.num_features))
        else:
            _replace_batchnorm(child)
    return module


IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class RgbEncoder(nn.Module):
    """DP's DiffusionRgbEncoder with ``vision_backbone=resnet18``.

    Default (Chi et al. image config, LeRobot 0.4.x default): trained from scratch, every
    BatchNorm replaced by GroupNorm.  ``imagenet=True`` (LeRobot 0.6.x default): BatchNorm kept
    and inputs normalized with the ImageNet mean / std; ``weights`` is a torchvision
    ``resnet18-f37072fd.pth`` state dict to start from (omit it when a checkpoint is loaded next).
    """

    SCALE_LAYERS = (2, 4, 5, 6, 7)        # backbone children after relu (1/2), layer1..layer4 (1/4 .. 1/32)
    SCALE_CHANNELS = (64, 64, 128, 256, 512)

    def __init__(self, image_size=256, crop=224, keypoints=32, imagenet=False, weights=None):
        super().__init__()
        self.crop_size = crop
        backbone = torchvision.models.resnet18(weights=None)
        if weights:
            backbone.load_state_dict(torch.load(str(weights), map_location="cpu", weights_only=True))
        body = nn.Sequential(*(list(backbone.children())[:-2]))
        self.backbone = body if imagenet else _replace_batchnorm(body)
        self.imagenet = imagenet
        # Not persistent: constants, and checkpoints of the GroupNorm encoder keep loading strictly.
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.backbone.eval()                                   # the shape probe must not touch BN statistics
        with torch.no_grad():
            shape = self.backbone(torch.zeros(1, 3, crop, crop)).shape[1:]
        self.backbone.train()
        self.pool = SpatialSoftmax(tuple(shape), num_kp=keypoints)
        self.feature_dim = keypoints * 2
        self.out = nn.Linear(keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def crop(self, x):
        """Random crop for training, center crop for evaluation (one offset per batch, as torchvision's
        RandomCrop / CenterCrop on a batch).  Returns the crop and its (row, column) offset."""
        crop = self.crop_size
        if self.training:
            i = int(torch.randint(0, x.shape[-2] - crop + 1, (1,)))
            j = int(torch.randint(0, x.shape[-1] - crop + 1, (1,)))
        else:
            i, j = int(round((x.shape[-2] - crop) / 2.0)), int(round((x.shape[-1] - crop) / 2.0))
        return x[..., i:i + crop, j:j + crop], (i, j)

    def forward(self, x, maps=False):
        """x [B, 3, H, W] in [0, 1] -> [B, 64]; with ``maps`` also the five feature scales of the crop
        (``SCALE_CHANNELS``, strides 2 .. 32) and the crop offset."""
        x, offset = self.crop(x)
        if self.imagenet:
            x = (x - self.mean) / self.std
        scales = []
        for index, layer in enumerate(self.backbone):
            x = layer(x)
            if maps and index in self.SCALE_LAYERS:
                scales.append(x)
        vector = self.relu(self.out(torch.flatten(self.pool(x), start_dim=1)))
        return (vector, scales, offset) if maps else vector


def anchors_from_tokens(tokens, used, steps=(8, 10, 12)):
    """Anchor points from ``policy_maps.keypoint_tokens`` [B, P, 13*4 + 2 + 3]: each used keypoint now and
    at +8 / +16 frames (sequence steps 10, 12) -> pixel (x, y) in the 256 frame [B, P*S, 2], validity,
    role one-hot [B, P*S, 3] and time one-hot [B, P*S, len(steps)]."""
    size, count = tokens.shape[:2]
    per_step = tokens[..., :52].reshape(size, count, 13, 4)              # (dx, dy) cells / 4, similarity, valid
    present = (tokens[..., 52:54] + 1) / 2 * 32 - 0.5                      # cells
    role = tokens[..., 54:57]
    points, valid, roles, times = [], [], [], []
    for k, step in enumerate(steps):
        points.append((present + per_step[:, :, step, :2] * 4 + 0.5) * 8)
        valid.append(used & (per_step[:, :, step, 3] > 0.5))
        roles.append(role)
        time = torch.zeros(size, count, len(steps), device=tokens.device)
        time[..., k] = 1
        times.append(time)
    return torch.cat(points, 1), torch.cat(valid, 1), torch.cat(roles, 1), torch.cat(times, 1)


class AnchorSampler(nn.Module):
    """Spatially conditioned image features (SCDP-style) at the student's keypoints.

    The current frame's five ResNet scales are projected (1x1) and bilinearly sampled at the anchor
    points (crop pixel coordinates); per anchor an MLP over [scales, role, time]; masked mean and max
    over anchors -> [B, 2 * hidden].  Gradients reach the image backbone through the samples.
    """

    def __init__(self, scale_channels=RgbEncoder.SCALE_CHANNELS, width=32, hidden=128, roles=3, times=3):
        super().__init__()
        self.project = nn.ModuleList([nn.Conv2d(c, width, 1) for c in scale_channels])
        self.mlp = nn.Sequential(nn.Linear(width * len(scale_channels) + roles + times, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden))
        self.width = 2 * hidden

    def forward(self, scales, points, valid, role, time, crop):
        grid = ((points + 0.5) / crop * 2 - 1)[:, None]                    # [B, 1, N, 2]
        sampled = [F.grid_sample(project(scale), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
                   [:, :, 0].transpose(1, 2) for project, scale in zip(self.project, scales)]
        tokens = self.mlp(torch.cat(sampled + [role, time], -1))           # [B, N, hidden]
        mask = valid[..., None].float()
        mean = (tokens * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        top = tokens.masked_fill(~valid[..., None], -1e4).max(1).values
        top = torch.where(valid.any(1, keepdim=True), top, torch.zeros_like(top))
        return torch.cat((mean, top), -1)


def scale_gradient(value, scale):
    """Identity forward; the backward gradient is multiplied by ``scale`` ([B, 1])."""
    return value * scale + value.detach() * (1 - scale)


class RgbDiffusionPolicy(LatentDiffusionPolicy):
    def __init__(self, action_dim=7, state_dim=8, cameras=2, image_size=256, crop=224, dp=None,
                 rgb_imagenet=False, rgb_weights=None):
        nn.Module.__init__(self)
        self.dp = {**DP_DEFAULTS, **(dp or {})}
        self.action_dim = action_dim
        self.rgb = nn.ModuleList([RgbEncoder(image_size, crop, imagenet=rgb_imagenet, weights=rgb_weights)
                                  for _ in range(cameras)])
        per_step = state_dim + sum(encoder.feature_dim for encoder in self.rgb)
        self.unet = DiffusionConditionalUnet1d(
            action_dim, per_step * self.dp["n_obs_steps"],
            down_dims=tuple(self.dp["down_dims"]), kernel_size=self.dp["kernel_size"], n_groups=self.dp["n_groups"],
            diffusion_step_embed_dim=self.dp["diffusion_step_embed_dim"],
            use_film_scale_modulation=self.dp["use_film_scale_modulation"])
        self.scheduler = self._scheduler("DDPM")

    def condition(self, batch):
        images = batch["images"]                                   # [B, steps, cameras, 3, H, W] in [0, 1]
        size, steps = images.shape[:2]
        features = [encoder(images[:, :, c].flatten(0, 1)).view(size, steps, -1)
                    for c, encoder in enumerate(self.rgb)]
        return torch.cat([batch["state"]] + features, -1).flatten(1)


class RgbLatentDiffusionPolicy(RgbDiffusionPolicy):
    """Vanilla DP plus the VACE latents as an additional condition (the intended design).

    The image path is exactly ``RgbDiffusionPolicy``; the latent tokens go through
    ``FeatureEncoder`` and its pooled vector is appended to DP's conditioning.
    """

    def __init__(self, action_dim=7, state_dim=8, cameras=2, image_size=256, crop=224, tasks=10, dp=None,
                 encoder=None):
        from .model import FeatureEncoder

        super().__init__(action_dim, state_dim, cameras, image_size, crop, dp)
        self.encoder = FeatureEncoder(state_dim=state_dim, n_obs_steps=self.dp["n_obs_steps"], tasks=tasks,
                                      **(encoder or {}))
        per_step = state_dim + sum(e.feature_dim for e in self.rgb)
        self.unet = DiffusionConditionalUnet1d(
            action_dim, per_step * self.dp["n_obs_steps"] + self.encoder.width,
            down_dims=tuple(self.dp["down_dims"]), kernel_size=self.dp["kernel_size"], n_groups=self.dp["n_groups"],
            diffusion_step_embed_dim=self.dp["diffusion_step_embed_dim"],
            use_film_scale_modulation=self.dp["use_film_scale_modulation"])

    def condition(self, batch):
        return torch.cat((super().condition(batch), self.encoder(batch)), -1)


class MapEncoder(nn.Module):
    """A: one camera's image-aligned policy maps [B, C, 32, 32] -> [B, width].

    Small GroupNorm conv stack read the way DP reads images (spatial-softmax keypoints) plus
    a global average for what the maps contain.
    """

    def __init__(self, channels, width=128, keypoints=32, grid=32):
        super().__init__()
        layers, previous = [], channels
        for out in (64, 128, 128):
            layers += [nn.Conv2d(previous, out, 3, padding=1), nn.GroupNorm(out // 16, out), nn.ReLU()]
            previous = out
        self.net = nn.Sequential(*layers)
        self.pool = SpatialSoftmax((previous, grid, grid), num_kp=keypoints)
        self.out = nn.Sequential(nn.Linear(keypoints * 2 + previous, width), nn.ReLU())
        self.width = width

    def forward(self, maps):
        features = self.net(maps)
        return self.out(torch.cat((self.pool(features).flatten(1), features.mean((2, 3))), -1))


class TrackEncoder(nn.Module):
    """C: keypoint-track tokens of both cameras [B, cams, P, D] (+ use mask) -> [B, width].

    Per-token MLP, camera embedding, a small transformer over all keypoints and attention
    pooling into one vector.  A learned always-present token keeps empty windows defined.
    """

    def __init__(self, token_dim, cameras=2, width=128, layers=2, heads=4):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(token_dim, width), nn.GELU(), nn.Linear(width, width))
        self.camera = nn.Parameter(torch.zeros(cameras, width))
        self.empty = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.query = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.pool = nn.MultiheadAttention(width, heads, batch_first=True)
        self.width = width

    def forward(self, tokens, used):
        size = tokens.shape[0]
        x = self.embed(tokens) + self.camera[None, :, None]
        x = torch.cat((self.empty.expand(size, -1, -1), x.flatten(1, 2)), 1)
        padding = torch.cat((torch.zeros(size, 1, dtype=torch.bool, device=used.device), ~used.flatten(1)), 1)
        x = self.encoder(x, src_key_padding_mask=padding)
        return self.pool(self.query.expand(size, -1, -1), x, x, key_padding_mask=padding)[0][:, 0]


class TrunkReader(nn.Module):
    """The student's fine-tuned V-JEPA2 trunk of both cameras [B, cams, T=17, 16, 16, 1024] -> [B, out].

    Video-Former style (VPP): per camera and time step, learned queries attend over the 256
    spatial tokens; a transformer then runs over all cameras x 17 times x queries with time
    and camera embeddings, and learned output queries pool it.  All 17 time steps are read.
    """

    def __init__(self, dim=1024, width=128, queries=4, cameras=2, times=17, cells=256, heads=4, out=256):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.project = nn.Linear(dim, width)
        self.spatial_position = nn.Parameter(torch.randn(cells, width) * 0.02)
        self.time_position = nn.Parameter(torch.randn(times, width) * 0.02)
        self.camera = nn.Parameter(torch.zeros(cameras, width))
        self.spatial_queries = nn.Parameter(torch.randn(queries, width) * 0.02)
        self.spatial = nn.MultiheadAttention(width, heads, batch_first=True)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, 2)
        self.output_queries = nn.Parameter(torch.randn(queries, width) * 0.02)
        self.pool = nn.MultiheadAttention(width, heads, batch_first=True)
        self.out = nn.Sequential(nn.Linear(queries * width, out), nn.ReLU())
        self.width = out

    def forward(self, trunk):
        size, cameras, times = trunk.shape[:3]
        x = self.project(self.norm(trunk.float().flatten(3, 4)))                    # [B, cams, T, 256, width]
        x = (x + self.spatial_position).flatten(0, 2)
        queries = self.spatial_queries[None].expand(len(x), -1, -1)
        x = self.spatial(queries, x, x)[0].reshape(size, cameras, times, -1, x.shape[-1])
        x = x + self.time_position[None, None, :, None] + self.camera[None, :, None, None]
        x = self.temporal(x.flatten(1, 3))
        pooled = self.pool(self.output_queries[None].expand(size, -1, -1), x, x)[0]
        return self.out(pooled.flatten(1))


class MapsDiffusionPolicy(RgbDiffusionPolicy):
    """Vanilla DP (ResNet-18 per camera, unchanged) plus three condition branches:

    A  ``maps`` [B, cams, C, 32, 32]   image-aligned single-meaning channels (``policy_maps``)
    C  ``tracks`` [B, cams, P, D]       filtered keypoint tracks, late-fused into the condition
    J  ``trunk`` [B, cams, 17, 16, 16, 1024]  the student's V-JEPA2 trunk, all 17 time steps

    During training each branch is zeroed per sample with probability ``branch_dropout``
    (so the policy cannot lean on one branch only, and zeroing a branch at evaluation is a
    meaningful ablation).  ``batch["drop"]`` (set of "maps" / "tracks" / "trunk") zeroes
    branches at evaluation.

    ``anchors``: SCDP-style image features sampled at the student's keypoints (``AnchorSampler``),
    so the fine image scales around the gripper / target reach the UNet.  ``batch["grad_scale"]``
    ([B], training only): the three student branches' gradients are scaled per sample (GAP-style:
    smaller near grasp / release so the image path has to learn those transitions).
    """

    BRANCHES = ("maps", "tracks", "trunk")

    def __init__(self, map_channels, token_dim, action_dim=7, state_dim=8, cameras=2, image_size=256, crop=224,
                 dp=None, branch_dropout=0.1, rgb_imagenet=False, rgb_weights=None, anchors=False):
        super().__init__(action_dim, state_dim, cameras, image_size, crop, dp, rgb_imagenet, rgb_weights)
        self.maps = nn.ModuleList([MapEncoder(map_channels) for _ in range(cameras)])
        self.tracks = TrackEncoder(token_dim, cameras)
        self.trunk = TrunkReader(cameras=cameras)
        self.anchors = nn.ModuleList([AnchorSampler() for _ in range(cameras)]) if anchors else None
        self.branch_dropout = branch_dropout
        per_step = state_dim + sum(encoder.feature_dim for encoder in self.rgb)
        extra = sum(m.width for m in self.maps) + self.tracks.width + self.trunk.width
        if self.anchors is not None:
            extra += sum(a.width for a in self.anchors)
        self.unet = DiffusionConditionalUnet1d(
            action_dim, per_step * self.dp["n_obs_steps"] + extra,
            down_dims=tuple(self.dp["down_dims"]), kernel_size=self.dp["kernel_size"], n_groups=self.dp["n_groups"],
            diffusion_step_embed_dim=self.dp["diffusion_step_embed_dim"],
            use_film_scale_modulation=self.dp["use_film_scale_modulation"])

    def condition(self, batch):
        images = batch["images"]                                   # [B, steps, cameras, 3, H, W] in [0, 1]
        size, steps = images.shape[:2]
        per_step, anchored = [], []
        for c, encoder in enumerate(self.rgb):
            flat = images[:, :, c].flatten(0, 1)
            if self.anchors is None:
                vector = encoder(flat)
            else:
                vector, scales, (row, column) = encoder(flat, maps=True)
                current = [scale.view(size, steps, *scale.shape[1:])[:, -1] for scale in scales]
                points, valid, role, time = anchors_from_tokens(batch["tracks"][:, c], batch["track_used"][:, c])
                points = points - points.new_tensor([column, row])
                anchored.append(self.anchors[c](current, points, valid, role, time, encoder.crop_size))
            per_step.append(vector.view(size, steps, -1))
        rgb = torch.cat([batch["state"]] + per_step, -1).flatten(1)
        branches = {"maps": torch.cat([encoder(batch["maps"][:, c]) for c, encoder in enumerate(self.maps)], -1),
                    "tracks": self.tracks(batch["tracks"], batch["track_used"]),
                    "trunk": self.trunk(batch["trunk"])}
        drop = batch.get("drop", ())
        scale = batch.get("grad_scale")                            # [B]: phase-guided gradient scale (training)
        for name, value in branches.items():
            keep = torch.ones(len(value), 1, device=value.device)
            if self.training and self.branch_dropout:
                keep = (torch.rand(len(value), 1, device=value.device) >= self.branch_dropout).float()
            if name in drop:
                keep = torch.zeros_like(keep)
            if scale is not None:
                value = scale_gradient(value, scale.to(value.dtype)[:, None])
            branches[name] = value * keep
        return torch.cat([rgb] + [branches[n] for n in self.BRANCHES] + anchored, -1)
