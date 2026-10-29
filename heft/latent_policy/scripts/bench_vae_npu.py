#!/usr/bin/env python3
"""Single-NPU check: is a batched VAE.encode equal to per-video encode, and how fast?"""
import sys
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from torch_npu.contrib import transfer_to_npu  # noqa: F401
from torch import amp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent))

from vace_latent_distill import teacher as T

device = torch.device("npu", 0)
torch.npu.set_device(device)
source = T.local_scripts()
config_root = ROOT.parents[1] / "models/Wan2.1-VACE-1.3B"
vae = source.load_vae(config_root / "Wan2.1_VAE.pth", device)

B, FRAMES, SIZE = 4, 49, 512
torch.manual_seed(0)
videos = [torch.randn(3, FRAMES, SIZE, SIZE, device=device).clamp(-1, 1) for _ in range(B)]


def sync():
    torch.npu.synchronize()
    return time.monotonic()


def per_video():
    return vae.encode(videos)


def batched():
    with amp.autocast("cuda", dtype=vae.dtype):
        out = vae.model.encode(torch.stack(videos, 0), vae.scale).float()
    return list(out.unbind(0))


for name, fn in (("per_video", per_video), ("batched", batched)):
    fn()  # warm-up
    t = sync()
    out = fn()
    print(f"{name:10s} {sync() - t:6.2f}s for {B} videos", flush=True)
    if name == "per_video":
        ref = out
    else:
        for a, b in zip(ref, out):
            print("  max|diff| =", float((a - b).abs().max()), " ref max|x| =", float(a.abs().max()))
