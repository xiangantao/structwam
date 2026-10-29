"""Past-only V-JEPA2, late text fusion, and separately trainable structured heads."""

import importlib
import math
import sys
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import HEAD_TASKS, OBSERVED_FRAMES, ModelConfig, shared_key, timeline


def load_backbone(config, *, pretrained=True):
    root = Path(config.vjepa_root).resolve(strict=True)
    if not (root / "src/models/vision_transformer.py").is_file():
        raise ValueError(f"Not a V-JEPA2 checkout: {root}")
    sys.path.insert(0, str(root))
    module = importlib.import_module("src.models.vision_transformer")
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise RuntimeError("V-JEPA2 import resolved to a different source tree")
    from latent_video.classification.baselines.vjepa2.encoder import load_target_encoder

    backbone = module.vit_large(
        img_size=config.image_size,
        num_frames=34,
        patch_size=16,
        tubelet_size=2,
        uniform_power=True,
        use_rope=True,
    )
    if config.dim != backbone.embed_dim:
        raise ValueError("Official ViT-L requires dim=1024")
    if pretrained:
        load_target_encoder(backbone, config.vjepa_weights)
    return backbone.requires_grad_(True)


def position_field(times, grid, dim):
    yy, xx = torch.meshgrid(
        torch.arange(grid, device=times.device),
        torch.arange(grid, device=times.device),
        indexing="ij",
    )
    coordinates = (
        times[:, None, None].expand(-1, grid, grid) / 4,
        yy[None].expand(len(times), -1, -1),
        xx[None].expand(len(times), -1, -1),
    )
    width = dim // 6
    frequencies = 10000 ** (-torch.arange(width, device=times.device).float() / max(width, 1))
    values = []
    for coordinate in coordinates:
        phase = coordinate[..., None] * frequencies
        values.extend((phase.sin(), phase.cos()))
    field = torch.cat(values, -1)
    return F.pad(field, (0, dim - field.shape[-1]))


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, query, memory=None, valid=None, allowed=None):
        memory = query if memory is None else memory
        batch, count, dim = query.shape
        q = self.q(query).reshape(batch, count, self.heads, dim // self.heads).transpose(1, 2)
        k, v = (
            self.kv(memory)
            .reshape(batch, -1, 2, self.heads, dim // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        mask = valid[:, None, None, :] if valid is not None else None
        if allowed is not None:
            mask = allowed[:, None] if mask is None else mask & allowed[:, None]
        if mask is not None and mask.shape[2] == 1 and count > 1:
            # Ascend FlashAttention rejects [B,1,1,Skv]; broadcast to [B,1,Sq,Skv].
            mask = mask.expand(-1, -1, count, -1)
        result = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        return self.proj(result.transpose(1, 2).reshape(batch, count, dim))


class TextFusion(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.text_norm = nn.LayerNorm(dim)
        self.attention = Attention(dim, heads)
        self.scale = nn.Parameter(torch.full((dim,), 1e-3))

    def forward(self, visual, text, valid):
        return visual + self.scale * self.attention(self.norm(visual), self.text_norm(text), valid)


class GridBlock(nn.Module):
    """Shifted windows use a boundary mask; opposite image edges cannot wrap together."""

    def __init__(self, config, shift=False):
        super().__init__()
        dim, heads = config.dim, config.attention_heads
        self.window = config.window_size
        self.shift = self.window // 2 if shift else 0
        self.space_norm = nn.LayerNorm(dim)
        self.time_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.space_attention = Attention(dim, heads)
        self.time_attention = Attention(dim, heads)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * config.mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * config.mlp_ratio, dim),
        )

    def windows(self, field):
        batch, temporal, height, width, dim = field.shape
        w = self.window
        return (
            field.reshape(batch, temporal, height // w, w, width // w, w, dim)
            .permute(0, 1, 2, 4, 3, 5, 6)
            .reshape(-1, w * w, dim)
        )

    def forward(self, field, history=None):
        """``history`` [B, Th, H, W, D]: extra temporal-attention keys at the same cell."""
        batch, temporal, height, width, dim = field.shape
        original = field
        value = self.space_norm(field)
        allowed = None
        if self.shift:
            value = torch.roll(value, (-self.shift, -self.shift), (2, 3))
            y, x = torch.meshgrid(
                torch.arange(height, device=field.device),
                torch.arange(width, device=field.device),
                indexing="ij",
            )
            y = (y + self.shift) % height
            x = (x + self.shift) % width
            labels = (y >= self.shift).long() * 2 + (x >= self.shift).long()
            labels = self.windows(labels[None, None, :, :, None]).squeeze(-1)
            allowed = (labels[:, :, None] == labels[:, None, :]).repeat(batch * temporal, 1, 1)
        value = self.space_attention(self.windows(value), allowed=allowed)
        w = self.window
        value = (
            value.reshape(batch, temporal, height // w, width // w, w, w, dim)
            .permute(0, 1, 2, 4, 3, 5, 6)
            .reshape(batch, temporal, height, width, dim)
        )
        if self.shift:
            value = torch.roll(value, (self.shift, self.shift), (2, 3))
        field = original + value
        value = self.time_norm(field).permute(0, 2, 3, 1, 4).reshape(-1, temporal, dim)
        context = None
        if history is not None:
            past = self.time_norm(history).permute(0, 2, 3, 1, 4).reshape(-1, history.shape[1], dim)
            context = torch.cat((past, value), 1)
        value = (
            self.time_attention(value, context)
            .reshape(batch, height, width, temporal, dim)
            .permute(0, 3, 1, 2, 4)
        )
        field = field + value
        return field + self.ffn(self.ffn_norm(field))


class DecoderBlock(nn.Module):
    def __init__(self, config, shift):
        super().__init__()
        self.grid = GridBlock(config, shift)
        self.norm = nn.LayerNorm(config.dim)
        self.memory_norm = nn.LayerNorm(config.dim)
        self.visual_attention = Attention(config.dim, config.attention_heads)
        self.text = TextFusion(config.dim, config.attention_heads)

    def forward(self, field, memory, text, valid, history=None):
        field = self.grid(field, history)
        shape = field.shape
        flat = field.flatten(1, 3)
        flat = flat + self.visual_attention(self.norm(flat), self.memory_norm(memory))
        return self.text(flat, text, valid).reshape(shape)


def cell_grid(grid, device):
    ys, xs = torch.meshgrid(
        torch.arange(grid, device=device), torch.arange(grid, device=device), indexing="ij"
    )
    return torch.stack((xs, ys), -1).float()


def sample_cells(field, positions):
    """Bilinear [B, G, G, C] at per-slot cell positions [B, S, G, G, 2] (x, y) -> [B, S, G, G, C]."""
    batch, grid, _, channels = field.shape
    slots = positions.shape[1]
    normalized = ((positions.float() + 0.5) / grid * 2 - 1).reshape(batch * slots, grid, grid, 2)
    values = (
        field.permute(0, 3, 1, 2).float()[:, None].expand(-1, slots, -1, -1, -1)
        .reshape(batch * slots, channels, grid, grid)
    )
    sampled = F.grid_sample(values, normalized, mode="bilinear", padding_mode="border", align_corners=False)
    return sampled.reshape(batch, slots, channels, grid, grid).permute(0, 1, 3, 4, 2)


def warp(field, flow):
    """Backward-warp [B, G, G, D] by per-slot flow [B, S, G, G, 2] (dx, dy cells) -> [B, S, G, G, D]."""
    source = cell_grid(field.shape[1], field.device) - flow.float()  # where each target cell's content was
    return sample_cells(field, source).to(field.dtype)


def soft_velocity(previous, current, grid, radius, temperature):
    """Unit descriptors [B, cells, C] at t-1 and t -> (dx, dy) cells per step [B, G, G, 2].

    Soft-argmax K-K match of each cell at ``t`` within ``radius`` cells at ``t-1``;
    the velocity is defined where the content sits at ``t``.
    """
    size = 2 * radius + 1
    batch, cells, channels = current.shape
    windows = F.unfold(
        previous.transpose(1, 2).reshape(batch, channels, grid, grid), size, padding=radius
    ).reshape(batch, channels, size * size, cells)
    similarity = (current.transpose(1, 2)[:, :, None] * windows).sum(1)  # [B, window, cells]
    inside = F.unfold(torch.ones(1, 1, grid, grid, device=current.device), size, padding=radius)
    similarity = similarity.masked_fill(inside == 0, -torch.inf)
    weights = (similarity / temperature).softmax(1)
    offsets = torch.arange(size, device=current.device).float() - radius
    dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
    before = torch.einsum("bwn,wd->bnd", weights, torch.stack((dx.flatten(), dy.flatten()), -1))
    return (-before).reshape(batch, grid, grid, 2)  # content moved from t-1 to t


@torch.no_grad()
def splat_velocity(velocity, multiples, sharpness=4.0, empty=0.05):
    """Velocity of the content that lands on each cell after ``multiples`` [S] steps.

    ``velocity`` [B, G, G, 2] is defined where content sits; content at ``q`` lands on
    ``q + m v(q)`` and its velocity is bilinearly splatted there -> [B, S, G, G, 2].
    Moving content is weighted ``exp(sharpness |v|)`` so it occludes the static
    background; ``empty`` is a pseudo-weight of zero motion for cells nothing reaches.
    """
    batch, grid = velocity.shape[:2]
    scale = multiples.float().to(velocity.device)[None, :, None, None]
    moving = velocity.float().reshape(batch, 1, grid * grid, 2)
    target = cell_grid(grid, velocity.device).reshape(1, 1, -1, 2) + scale * moving
    corner, fraction = target.floor(), target - target.floor()
    importance = torch.exp(sharpness * moving.norm(dim=-1))
    total = torch.zeros(batch, len(multiples), grid * grid, 3, device=velocity.device)
    for dx in (0, 1):
        for dy in (0, 1):
            x, y = corner[..., 0] + dx, corner[..., 1] + dy
            weight = (
                (fraction[..., 0] if dx else 1 - fraction[..., 0])
                * (fraction[..., 1] if dy else 1 - fraction[..., 1])
                * importance
                * ((x >= 0) & (x < grid) & (y >= 0) & (y < grid))
            )
            index = (y.clamp(0, grid - 1) * grid + x.clamp(0, grid - 1)).long()
            values = torch.cat((weight[..., None] * moving.expand_as(target), weight[..., None]), -1)
            total.scatter_add_(2, index[..., None].expand(-1, -1, -1, 3), values)
    return (total[..., :2] / (total[..., 2:] + empty)).reshape(batch, len(multiples), grid, grid, 2)


def extrapolate(velocity, steps):
    """Constant-velocity backward flow on the target grid.

    ``velocity`` is defined where content sits at the last observed time; the base
    field is one native step later, so content reaching target ``p`` in a slot ``s``
    steps after the base has travelled ``s + 1`` steps and is sampled from the base
    at ``p - s v``.  Reading ``v`` at the target itself would give the velocity of
    whatever is behind a moving object.
    """
    steps = steps.to(velocity.device)
    return steps.float()[None, :, None, None, None] * splat_velocity(velocity, steps + 1)


class FutureMotion(nn.Module):
    """Per-slot future flow in cells for ``warp``.

    Prior: each cell's soft-argmax K-K match from the last observed time back to the
    previous one, on the student's own detached L15H7 K field, gives a velocity per
    native step; slot ``t`` extrapolates it ``steps[t]`` steps (``extrapolate``), scaled
    by a learned ``gain``.  A zero-initialized MLP
    on the last decoded observed field adds a residual, so training starts at the prior.
    All observed inputs are detached: future losses never retrain the observed branch.
    """

    def __init__(self, config, slots):
        super().__init__()
        self.radius = config.future_motion_radius
        self.temperature = config.future_motion_temperature
        self.gain = nn.Parameter(torch.ones(1))
        self.norm = nn.LayerNorm(config.dim)
        self.residual = nn.Sequential(
            nn.Linear(config.dim + 2, 256), nn.GELU(), nn.Linear(256, 2 * slots)
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def velocity(self, previous, current, grid):
        return soft_velocity(previous, current, grid, self.radius, self.temperature)

    def forward(self, field, previous, current, steps):
        """``field`` [B, G, G, D] last decoded observed; ``steps`` [S] native steps ahead per slot."""
        batch, grid = field.shape[:2]
        velocity = self.velocity(previous.detach().float(), current.detach().float(), grid)
        residual = self.residual(torch.cat((self.norm(field.detach()).float(), velocity), -1))
        residual = residual.reshape(batch, grid, grid, len(steps), 2).permute(0, 3, 1, 2, 4)
        return self.gain * extrapolate(velocity, steps) + residual


class ResidualProjection(nn.Module):
    def __init__(self, dim, channels, ratio):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.skip = nn.Linear(dim, channels)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * ratio), nn.GELU(), nn.Linear(dim * ratio, channels)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, field):
        return self.skip(field) + self.mlp(self.norm(field))


class StructuredHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(
            [GridBlock(config, bool(i % 2)) for i in range(config.head_layers)]
        )
        count = 1 if config.task == "depth" else config.channels
        self.primary = ResidualProjection(config.dim, count, config.mlp_ratio)
        self.secondary = (
            ResidualProjection(config.dim, config.channels, config.mlp_ratio)
            if config.task in {"tracking", "semantic"}
            else None
        )

    def forward(self, field, text, valid):
        for block in self.blocks:
            field = (
                checkpoint(block, field, use_reentrant=False)
                if self.training and self.config.activation_checkpointing
                else block(field)
            )
        # Keep the native [time, spatial-cell, channel] contract.  Only the two
        # spatial axes are flattened; the channel axis remains the Q/K width.
        primary = self.primary(field).flatten(2, 3)
        if self.config.task.startswith("seg_"):
            return {"seg_k": F.normalize(primary, dim=-1)}
        if self.config.task == "depth":
            return {"depth": primary.squeeze(-1)}
        if self.config.task == "tracking":
            secondary = self.secondary(field).flatten(2, 3)
            return {
                "track_q": F.normalize(primary, dim=-1),
                "track_k": F.normalize(secondary, dim=-1),
            }
        text_key = self.secondary(text)
        logits = torch.einsum("btgc,blc->btgl", primary.float(), text_key.float()) / math.sqrt(
            self.config.channels
        )
        logits = logits.masked_fill(~valid[:, None, None, :], -torch.inf)
        return {"text_attention": logits.softmax(-1)}


class StructuredStudent(nn.Module):
    def __init__(self, config: ModelConfig, backbone=None, *, pretrained=True):
        super().__init__()
        config.validate()
        self.config = config
        self.backbone = (
            load_backbone(config, pretrained=pretrained) if backbone is None else backbone
        )
        self.backbone.requires_grad_(True)
        if config.language_layers > len(self.backbone.blocks):
            raise ValueError("Too many language fusion layers for the backbone")
        self.text_projection = nn.Sequential(
            nn.LayerNorm(config.text_dim), nn.Linear(config.text_dim, config.dim)
        )
        self.language = nn.ModuleList(
            [TextFusion(config.dim, config.attention_heads) for _ in range(config.language_layers)]
        )
        self.observed_blocks = nn.ModuleList(
            [DecoderBlock(config, bool(i % 2)) for i in range(config.decoder_layers)]
        )
        self.future_blocks = nn.ModuleList(
            [
                DecoderBlock(config, bool(i % 2))
                for i in range(config.future_layers or config.decoder_layers)
            ]
        )
        future = timeline()["future"]
        # Native slot of the last observed frame; later slots are true future.
        self.future_present = int(future["valid"].nonzero()[0])
        self.future_motion = (
            FutureMotion(config, len(future["native_time"]))
            if config.future_init == "warp"
            else None
        )
        self.branch_embedding = nn.Parameter(torch.zeros(2, config.dim))
        # "shared" adds one vector to every copy of the last observed frame; with
        # "per_time" each future readout slot starts from its own learned query.
        shape = (
            (len(timeline()["future"]["native_time"]), config.dim)
            if config.future_query == "per_time"
            else (config.dim,)
        )
        self.future_query = nn.Parameter(torch.empty(shape))
        nn.init.normal_(self.future_query, std=0.02)
        if config.learnable_position_scale:
            self.position_scale = nn.Parameter(torch.full((1,), float(config.position_scale)))
        if config.task == "shared":
            self.heads = nn.ModuleDict(
                {task: StructuredHead(replace(config, task=task)) for task in HEAD_TASKS}
            )
        else:
            self.head = StructuredHead(config)
        self.register_buffer("rgb_mean", torch.tensor((0.485, 0.456, 0.406)).reshape(1, 3, 1, 1, 1))
        self.register_buffer("rgb_std", torch.tensor((0.229, 0.224, 0.225)).reshape(1, 3, 1, 1, 1))

    def encode_past(self, observed_rgb, text, valid):
        if (
            observed_rgb.dtype != torch.uint8
            or observed_rgb.ndim != 5
            or observed_rgb.shape[1:3] != (OBSERVED_FRAMES, 3)
        ):
            raise ValueError("Student accepts exactly33 past uint8 frames [B,33,3,H,W]")
        if observed_rgb.shape[-2:] != (self.config.image_size, self.config.image_size):
            raise ValueError("Student RGB size differs from model config")
        # Tubelets have width2. Repeat the LAST past frame so the 33rd frame is not dropped.
        pixels = (
            torch.cat((observed_rgb, observed_rgb[:, -1:]), 1).permute(0, 2, 1, 3, 4).float() / 255
        )
        pixels = (pixels - self.rgb_mean) / self.rgb_std
        value = self.backbone.patch_embed(pixels)
        temporal, height, width = pixels.shape[2] // 2, pixels.shape[3] // 16, pixels.shape[4] // 16
        if value.shape[1:] != (temporal * height * width, self.config.dim):
            raise ValueError("Unexpected V-JEPA2 T/H/W token order")
        first_language_layer = len(self.backbone.blocks) - len(self.language)
        for index, block in enumerate(self.backbone.blocks):

            def run(x, block=block):
                return block(
                    x, mask=None, attn_mask=None, T=temporal, H_patches=height, W_patches=width
                )

            value = (
                checkpoint(run, value, use_reentrant=False)
                if self.training and self.config.activation_checkpointing
                else run(value)
            )
            if index >= first_language_layer:
                fusion = self.language[index - first_language_layer]
                value = (
                    checkpoint(fusion, value, text, valid, use_reentrant=False)
                    if self.training and self.config.activation_checkpointing
                    else fusion(value, text, valid)
                )
        return self.backbone.norm(value).reshape(
            len(pixels), temporal, height, width, self.config.dim
        )

    def positions(self, times, dtype):
        """Fixed sinusoidal readout positions, optionally scaled (learned or constant)."""
        field = position_field(times, self.config.grid_size, self.config.dim).to(dtype)
        if self.config.learnable_position_scale:
            return field * self.position_scale.to(dtype)
        if self.config.position_scale != 1.0:
            return field * self.config.position_scale
        return field

    def forward(self, observed_rgb, text_embeddings, text_mask, observed_heads=None):
        """No future RGB/latent/teacher argument exists in the student API.

        ``observed_heads`` (shared task only) limits which heads read the observed branch;
        consumers that only need some observed outputs skip the rest.  ``None`` reads all.
        """
        if text_embeddings.ndim != 3 or text_embeddings.shape[-1] != self.config.text_dim:
            raise ValueError("Invalid text embedding dimensions")
        if (
            text_mask.dtype != torch.bool
            or text_mask.shape != text_embeddings.shape[:2]
            or not text_mask.any(-1).all()
        ):
            raise ValueError("Each instruction needs at least one valid text token")
        if observed_rgb.shape[0] != text_embeddings.shape[0]:
            raise ValueError("RGB and instruction batch sizes differ")
        text = self.text_projection(text_embeddings)
        visual = self.encode_past(observed_rgb, text, text_mask)
        memory = visual.flatten(1, 3)
        times = timeline(visual.device)
        grid = self.config.grid_size
        source = visual.permute(0, 4, 1, 2, 3)
        # This initializes the learned readout grid; it does not add native image detail.
        observed_times = times["observed"]["nominal_rgb_time"].clamp(0, 32).float()
        spatial = torch.linspace(-1, 1, grid, device=visual.device)
        zz, yy, xx = torch.meshgrid(
            (observed_times - 0.5).clamp(0, 32) / 16 - 1, spatial, spatial, indexing="ij"
        )
        sampling = torch.stack((xx, yy, zz), -1)[None].expand(len(visual), -1, -1, -1, -1)
        # CUDA grid_sample requires matching dtypes; perform the small resampling in float32.
        observed = (
            F.grid_sample(source.float(), sampling, mode="bilinear", align_corners=True)
            .permute(0, 2, 3, 4, 1)
            .to(visual.dtype)
        )
        observed = (
            observed + self.positions(times["observed"]["nominal_rgb_time"], visual.dtype)[None]
        )
        observed = observed + self.branch_embedding[0]
        last = F.interpolate(
            source[:, :, -1].float(), size=(grid, grid), mode="bilinear", align_corners=False
        )
        last = last.permute(0, 2, 3, 1).to(visual.dtype)
        output = {"trunk": visual}  # V-JEPA2 trunk after language fusion [B, 17, 16, 16, D], shared by both branches
        observed = self.decode(observed, self.observed_blocks, memory, text, text_mask)
        if observed_heads is not None and self.future_motion is not None:
            observed_heads = set(observed_heads) | {"seg_l15h7"}  # the warp prior reads it
        output["observed"] = self.readout(observed, text, text_mask, observed_heads)
        if self.future_motion is not None:
            # Slot t sits (t - present) native steps after the last observed frame; the
            # observed branch's last two valid times give the per-step velocity.
            steps = (times["future"]["native_time"] - self.future_present).clamp(min=0)
            k_field = output["observed"]["seg_k_l15h7"]
            flow = self.future_motion(observed[:, -1], k_field[:, -2], k_field[:, -1], steps)
            start = warp(last, flow)
        else:
            start = last[:, None]
        query = self.future_query
        if query.ndim == 2:  # per_time: [13, dim] -> one query per future slot
            query = query[:, None, None]
        future = start + query + self.branch_embedding[1]
        future = future + self.positions(times["future"]["nominal_rgb_time"], visual.dtype)[None]
        history = None
        if self.config.future_context == "observed":
            history = observed[:, times["observed"]["valid"]].detach()
        future = self.decode(future, self.future_blocks, memory, text, text_mask, history)
        output["future"] = self.readout(future, text, text_mask)
        output["timeline"] = times
        return output

    def decode(self, field, blocks, memory, text, text_mask, history=None):
        for block in blocks:
            field = (
                checkpoint(block, field, memory, text, text_mask, history, use_reentrant=False)
                if self.training and self.config.activation_checkpointing
                else block(field, memory, text, text_mask, history)
            )
        return field

    def readout(self, field, text, text_mask, heads=None):
        if self.config.task == "shared":
            return {
                shared_key(task, key): value
                for task, head in self.heads.items()
                if heads is None or task in heads
                for key, value in head(field, text, text_mask).items()
            }
        return self.head(field, text, text_mask)
