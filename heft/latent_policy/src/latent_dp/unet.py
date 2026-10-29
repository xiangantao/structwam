# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Diffusion Policy 1D conditional UNet, copied from LeRobot 0.6.2
(``lerobot/policies/diffusion/modeling_diffusion.py``) so it runs without LeRobot's
package dependencies.  Only the configuration object is replaced by explicit arguments
and ``einops`` by ``permute``; layers, order and initialization are unchanged.
"""

import math

import torch
from torch import Tensor, nn


class DiffusionSinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class DiffusionConv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class DiffusionConditionalResidualBlock1d(nn.Module):
    """ResNet style 1D convolutional block with FiLM modulation for conditioning."""

    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8,
                 use_film_scale_modulation=False):
        super().__init__()
        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels
        self.conv1 = DiffusionConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))
        self.conv2 = DiffusionConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        out = self.conv1(x)
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            out = out + cond_embed
        out = self.conv2(out)
        return out + self.residual_conv(x)


class DiffusionConditionalUnet1d(nn.Module):
    """A 1D convolutional UNet with FiLM modulation for conditioning (LeRobot / Chi et al.)."""

    def __init__(self, action_dim, global_cond_dim, down_dims=(512, 1024, 2048), kernel_size=5, n_groups=8,
                 diffusion_step_embed_dim=128, use_film_scale_modulation=True):
        super().__init__()
        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        cond_dim = diffusion_step_embed_dim + global_cond_dim
        in_out = [(action_dim, down_dims[0])] + list(zip(down_dims[:-1], down_dims[1:], strict=True))
        common = {"cond_dim": cond_dim, "kernel_size": kernel_size, "n_groups": n_groups,
                  "use_film_scale_modulation": use_film_scale_modulation}
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                DiffusionConditionalResidualBlock1d(dim_in, dim_out, **common),
                DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common),
                nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
            ]))
        self.mid_modules = nn.ModuleList([
            DiffusionConditionalResidualBlock1d(down_dims[-1], down_dims[-1], **common),
            DiffusionConditionalResidualBlock1d(down_dims[-1], down_dims[-1], **common),
        ])
        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                DiffusionConditionalResidualBlock1d(dim_in * 2, dim_out, **common),
                DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common),
                nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
            ]))
        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(down_dims[0], down_dims[0], kernel_size=kernel_size),
            nn.Conv1d(down_dims[0], action_dim, 1),
        )

    def forward(self, x: Tensor, timestep: Tensor, global_cond=None) -> Tensor:
        """x [B, T, action_dim], timestep [B], global_cond [B, global_cond_dim] -> [B, T, action_dim]."""
        x = x.permute(0, 2, 1)
        timesteps_embed = self.diffusion_step_encoder(timestep)
        global_feature = timesteps_embed if global_cond is None else torch.cat([timesteps_embed, global_cond], -1)
        skips = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet2(resnet(x, global_feature), global_feature)
            skips.append(x)
            x = downsample(x)
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, skips.pop()), dim=1)
            x = resnet2(resnet(x, global_feature), global_feature)
            x = upsample(x)
        return self.final_conv(x).permute(0, 2, 1)
