"""Diffusion Policy (Chi et al. / LeRobot UNet) conditioned on post-processed VACE features.

The observation encoder replaces DP's ResNet image encoder.  Every input becomes a
``width``-dim token: K blocks (one per camera and slot), point tracks (one per point,
sharing its block's slot embedding), the task's selected words, the two proprio states
and the task id; a type and a camera embedding are added; invalid blocks / points / words
are masked.  Two Transformer layers mix the tokens and a learned query pools them.  As in
DP, the conditioning vector is that feature concatenated with the ``n_obs_steps`` states.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .unet import DiffusionConditionalUnet1d

DP_DEFAULTS = {  # Chi et al. 2023 / LeRobot DiffusionConfig, LIBERO action chunking
    "n_obs_steps": 2, "horizon": 16, "n_action_steps": 8,
    "down_dims": (512, 1024, 2048), "kernel_size": 5, "n_groups": 8, "diffusion_step_embed_dim": 128,
    "use_film_scale_modulation": True, "num_train_timesteps": 100, "beta_schedule": "squaredcos_cap_v2",
    "beta_start": 1e-4, "beta_end": 0.02, "prediction_type": "epsilon", "clip_sample": True,
    "clip_sample_range": 1.0,
}


def mlp(inputs, width):
    return nn.Sequential(nn.Linear(inputs, width), nn.GELU(), nn.Linear(width, width))


class FeatureEncoder(nn.Module):
    def __init__(self, object_dim=42, track_dim=36, prompt_dim=66, state_dim=8, n_obs_steps=2, tasks=10,
                 cameras=2, blocks=8, width=256, layers=2, heads=8, dropout=0.1):
        super().__init__()
        self.objects = mlp(object_dim, width)
        self.tracks = mlp(track_dim, width)
        self.words = mlp(prompt_dim, width)
        self.state = mlp(state_dim * n_obs_steps, width)
        self.task = nn.Embedding(tasks, width)
        self.kind = nn.Embedding(5, width)          # block, point, word, state, task
        self.camera = nn.Embedding(cameras, width)
        self.slot = nn.Embedding(blocks, width)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.query = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.pool = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.width = width

    def forward(self, batch):
        objects, tracks, owner = batch["objects"], batch["tracks"], batch["owner"]
        size, cameras, blocks = objects.shape[:3]
        camera = self.camera.weight[:cameras][None, :, None]
        block = self.objects(objects) + self.kind.weight[0] + camera + self.slot.weight[:blocks][None, None]
        point = (self.tracks(tracks.flatten(-2)) + self.kind.weight[1] + camera
                 + self.slot(owner.clamp(min=0).long()))
        word = self.words(batch["prompt"]) + self.kind.weight[2]
        state = self.state(batch["state"].flatten(1))[:, None] + self.kind.weight[3]
        task = self.task(batch["task"])[:, None] + self.kind.weight[4]
        tokens = torch.cat((block.flatten(1, 2), point.flatten(1, 2), word, state, task), 1)
        valid = torch.cat((batch["object_valid"].flatten(1), (owner >= 0).flatten(1), batch["word_valid"],
                           torch.ones(size, 2, dtype=torch.bool, device=objects.device)), 1)
        mixed = self.encoder(tokens, src_key_padding_mask=~valid)
        pooled, _ = self.pool(self.query.expand(size, -1, -1), mixed, mixed, key_padding_mask=~valid)
        return self.norm(pooled[:, 0])


class LatentDiffusionPolicy(nn.Module):
    def __init__(self, action_dim=7, state_dim=8, tasks=10, encoder=None, dp=None):
        super().__init__()
        self.dp = {**DP_DEFAULTS, **(dp or {})}
        self.action_dim = action_dim
        self.encoder = FeatureEncoder(state_dim=state_dim, n_obs_steps=self.dp["n_obs_steps"], tasks=tasks,
                                      **(encoder or {}))
        self.unet = DiffusionConditionalUnet1d(
            action_dim, self.encoder.width + state_dim * self.dp["n_obs_steps"],
            down_dims=tuple(self.dp["down_dims"]), kernel_size=self.dp["kernel_size"], n_groups=self.dp["n_groups"],
            diffusion_step_embed_dim=self.dp["diffusion_step_embed_dim"],
            use_film_scale_modulation=self.dp["use_film_scale_modulation"])
        self.scheduler = self._scheduler("DDPM")

    def _scheduler(self, kind):
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

        options = {k: self.dp[k] for k in ("num_train_timesteps", "beta_start", "beta_end", "beta_schedule",
                                           "clip_sample", "clip_sample_range", "prediction_type")}
        return (DDPMScheduler if kind == "DDPM" else DDIMScheduler)(**options)

    def condition(self, batch):
        return torch.cat((self.encoder(batch), batch["state"].flatten(1)), -1)

    def forward(self, noisy, timesteps, batch):
        """Denoiser call (the DDP entry point): prediction for ``noisy`` at ``timesteps``."""
        return self.unet(noisy, timesteps, global_cond=self.condition(batch))

    def loss(self, batch):
        """DP training objective: predict the noise added to the normalized action chunk."""
        trajectory = batch["action"]
        noise = torch.randn_like(trajectory)
        steps = torch.randint(0, self.dp["num_train_timesteps"], (len(trajectory),), device=trajectory.device).long()
        noisy = self.scheduler.add_noise(trajectory, noise, steps)
        prediction = self(noisy, steps, batch)
        target = noise if self.dp["prediction_type"] == "epsilon" else trajectory
        return F.mse_loss(prediction, target)

    @torch.no_grad()
    def sample(self, batch, scheduler="DDPM", steps=None, generator=None):
        """Normalized action chunk [B, horizon, action_dim] by reverse diffusion."""
        cond = self.condition(batch)
        noise_scheduler = self.scheduler if scheduler == "DDPM" else self._scheduler(scheduler)
        noise_scheduler.set_timesteps(steps or self.dp["num_train_timesteps"])
        sample = torch.randn((len(cond), self.dp["horizon"], self.action_dim), device=cond.device,
                             generator=generator)
        for t in noise_scheduler.timesteps:
            output = self.unet(sample, torch.full((len(cond),), int(t), device=cond.device, dtype=torch.long), cond)
            sample = noise_scheduler.step(output, t, sample, generator=generator).prev_sample
        return sample

    def executed(self, chunk):
        """The ``n_action_steps`` actions DP executes from a chunk (starting at the current step)."""
        start = self.dp["n_obs_steps"] - 1
        return chunk[:, start : start + self.dp["n_action_steps"]]


class EMA:
    """DP's EMA of weights: decay = 1 - (1 + step) ** -0.75, capped at 0.9999 (diffusers EMAModel)."""

    def __init__(self, model, power=0.75, max_decay=0.9999):
        import copy

        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.power, self.max_decay, self.step = power, max_decay, 0

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        decay = min(max(1 - (1 + self.step) ** -self.power, 0.0), self.max_decay)
        for shadow, value in zip(self.model.parameters(), model.parameters(), strict=True):
            shadow.mul_(decay).add_(value.detach(), alpha=1 - decay)
        for shadow, value in zip(self.model.buffers(), model.buffers(), strict=True):
            shadow.copy_(value)
