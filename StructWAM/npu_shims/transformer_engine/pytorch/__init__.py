"""``transformer_engine.pytorch`` stand-in: RMSNorm and the submodules Cosmos imports."""

import os

import torch
from torch import nn

try:  # Ascend fused kernels (same results as the torch path up to bf16 rounding, ~20% faster DiT step)
    import torch_npu as _npu
except ImportError:  # pragma: no cover - non-Ascend hosts
    _npu = None
FUSED = _npu is not None and os.environ.get("TE_SHIM_FUSED", "1") != "0"

from transformer_engine.pytorch import attention, distributed, module  # noqa: F401


class RMSNorm(nn.Module):
    """TE ``RMSNorm``: ``x / sqrt(mean(x^2) + eps) * gamma`` over the last dims, in float32.

    ``zero_centered_gamma`` stores ``gamma - 1`` like TE.  The ``_extra_state`` entry of TE
    checkpoints (FP8 / RNG metadata) is accepted and ignored.
    """

    def __init__(self, normalized_shape, eps: float = 1e-5, sequence_parallel=None, params_dtype=None,
                 zero_centered_gamma: bool = False, device=None, **_):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = float(eps)
        self.zero_centered_gamma = bool(zero_centered_gamma)
        device = None if device == "cuda" else device
        self.weight = nn.Parameter(torch.empty(self.normalized_shape, dtype=params_dtype or torch.float32,
                                               device=device))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.zero_centered_gamma:
                self.weight.zero_()
            else:
                self.weight.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if FUSED and x.device.type == "npu" and not self.zero_centered_gamma and len(self.normalized_shape) == 1:
            return _npu.npu_rms_norm(x, self.weight.to(x.dtype), epsilon=self.eps)[0]
        dims = tuple(range(-len(self.normalized_shape), 0))
        x32 = x.float()
        y = x32 * torch.rsqrt(x32.pow(2).mean(dim=dims, keepdim=True) + self.eps)
        gamma = self.weight.float() + 1.0 if self.zero_centered_gamma else self.weight.float()
        return (y * gamma).to(x.dtype)

    def get_extra_state(self) -> torch.Tensor:
        return torch.empty(0, dtype=torch.uint8)

    def set_extra_state(self, state) -> None:
        return None

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}, zero_centered_gamma={self.zero_centered_gamma}"
