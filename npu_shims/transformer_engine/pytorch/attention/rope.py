"""TE rotary position embedding (``apply_rotary_pos_emb``): torch, or ``torch_npu.npu_rotary_mul`` on NPU."""

import os
from typing import Optional

import torch

try:
    import torch_npu as _npu
except ImportError:  # pragma: no cover - non-Ascend hosts
    _npu = None
FUSED = _npu is not None and os.environ.get("TE_SHIM_FUSED", "1") != "0"


def _rotate_half(x: torch.Tensor, interleaved: bool) -> torch.Tensor:
    if not interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_pos_emb(
    t: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "sbhd",
    start_positions: Optional[torch.Tensor] = None,
    interleaved: bool = False,
    fused: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    cp_size: int = 1,
    cp_rank: int = 0,
    **_,
) -> torch.Tensor:
    """``t`` [s, b, h, d] (``sbhd``) or [b, s, h, d] (``bshd``); ``freqs`` [max_s, 1, 1, rot_d].

    ``out = t * cos(freqs) + rotate_half(t) * sin(freqs)`` on the first ``rot_d`` channels, the rest
    passed through; computed in float32 and cast back (as TE's fused kernel).  ``fused`` only
    selects TE's kernel there and does not change the result.
    """
    if tensor_format not in ("sbhd", "bshd"):
        raise NotImplementedError(f"torch RoPE shim supports sbhd / bshd, not {tensor_format!r}")
    if cu_seqlens is not None or cp_size != 1 or start_positions is not None:
        raise NotImplementedError("torch RoPE shim: packed sequences, context parallel and start offsets "
                                  "are not supported")
    seq_dim = 1 if tensor_format == "bshd" else 0
    length = t.shape[seq_dim]
    if length > freqs.shape[0]:
        raise ValueError(f"sequence length {length} exceeds the rotary table ({freqs.shape[0]})")
    freqs = freqs[:length].float()
    if tensor_format == "bshd":
        freqs = freqs.transpose(0, 1)                      # [s, 1, 1, d] -> [1, s, 1, d]
    rot = freqs.shape[-1]
    if FUSED and t.device.type == "npu" and not interleaved and rot == t.shape[-1]:
        # Same rotate-half formula as below (checked against it on the Cosmos DiT: identical
        # denoising error, velocity difference at bf16 rounding).
        return _npu.npu_rotary_mul(t, torch.cos(freqs).to(t.dtype), torch.sin(freqs).to(t.dtype))
    x, passthrough = t[..., :rot].float(), t[..., rot:]
    out = x * torch.cos(freqs) + _rotate_half(x, interleaved) * torch.sin(freqs)
    return torch.cat((out.to(t.dtype), passthrough), dim=-1)
