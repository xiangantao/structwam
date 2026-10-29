"""Structure expert: the third stream of the StructWAM tri-system.

Same per-layer contract and blocks as the understanding expert (``und_expert.py``): per video
DiT layer it emits Q/K/V in the video head space for the joint attention and updates its own
tokens afterwards.  Only the input differs — instead of projecting a frozen VLM's hidden state,
it embeds the VACE student's structure features (``openwam.structure.tokens``).
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from openwam.model.architectures.tri_system.und_expert import (
    UnderstandingExpert,
    UnderstandingExpertBlock,
    UnderstandingExpertConfig,
)
from openwam.structure.tokens import GROUPS, StructureTokenizer


@dataclass
class StructureExpertConfig:
    dim: int = 512
    ffn_dim: int = 2048
    num_layers: int = 30
    eps: float = 1e-5
    cameras: Sequence[str] = ("head", "wrist")
    map_channels: int = 37
    word_slots: int = 5
    groups: Sequence[str] = GROUPS


@dataclass
class StructureState:
    """Driver-facing state; field names match ``UnderstandingState`` (``und_tokens`` / ``und_mask``)."""

    und_tokens: Tensor
    und_mask: Optional[Tensor] = None
    pooled: Optional[Tensor] = None


class StructureExpert(UnderstandingExpert):
    """Understanding-expert blocks over structure tokens (pre/post-attention hooks inherited)."""

    def __init__(self, cfg: StructureExpertConfig, wan_dim: int, wan_num_heads: int):
        nn.Module.__init__(self)
        block_cfg = UnderstandingExpertConfig(dim=cfg.dim, ffn_dim=cfg.ffn_dim, num_layers=cfg.num_layers,
                                              vlm_input_dim=cfg.dim, vlm_projector_type="linear", eps=cfg.eps)
        self.cfg = block_cfg
        self.structure_cfg = cfg
        self._num_heads = wan_num_heads
        self._head_dim = wan_dim // wan_num_heads
        self.tokenizer = StructureTokenizer(cfg.dim, cfg.cameras, cfg.map_channels, cfg.word_slots, cfg.groups)
        self.blocks = nn.ModuleList(
            [UnderstandingExpertBlock(block_cfg, wan_dim, wan_num_heads) for _ in range(cfg.num_layers)]
        )

    def prepare_state(self, features: dict, *, batch: int, dtype: Optional[torch.dtype] = None) -> StructureState:
        out = self.tokenizer(features, batch, dtype=dtype)
        return StructureState(und_tokens=out["tokens"], und_mask=out["mask"], pooled=out["pooled"])


__all__ = ["StructureExpert", "StructureExpertConfig", "StructureState"]
