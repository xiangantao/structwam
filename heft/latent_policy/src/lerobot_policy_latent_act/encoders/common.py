"""Shared pixel contract: the two backbones see exactly the same RGB frames."""

import hashlib
from pathlib import Path

import torch


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_local_file(path: str, sha256: str = "") -> Path:
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"Expected a local file: {resolved}")
    if sha256 and file_sha256(resolved) != sha256:
        raise ValueError(f"Checkpoint/mask checksum mismatch: {resolved}")
    return resolved


def rgb_uint8(images: list[torch.Tensor], frames: int, size: int) -> list[torch.Tensor]:
    result = []
    batch_size = images[0].shape[0]
    for image in images:
        if tuple(image.shape) != (batch_size, frames, 3, size, size):
            raise ValueError(f"Expected [B,{frames},3,{size},{size}], got {tuple(image.shape)}")
        if image.dtype == torch.uint8:
            result.append(image.detach().to("cpu").contiguous())
            continue
        if not image.is_floating_point():
            raise TypeError("Images must be uint8 or unnormalized floating point RGB in [0,1]")
        image = image.detach().to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError(
                "Images must be finite and in [0,1]; disable visual mean/std normalization"
            )
        result.append(image.mul(255).round().to(torch.uint8).contiguous())
    return result
