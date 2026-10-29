#!/usr/bin/env python3
"""Operator-level profile of one VAE.encode on a single NPU (diagnosis only)."""
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from torch_npu.contrib import transfer_to_npu  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent))
from vace_latent_distill import teacher as T

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/op_prof_vae"
device = torch.device("npu", 0)
torch.npu.set_device(device)
vae = T.local_scripts().load_vae(ROOT.parents[1] / "models/Wan2.1-VACE-1.3B/Wan2.1_VAE.pth", device)
video = [torch.randn(3, 49, 512, 512, device=device).clamp(-1, 1)]
vae.encode(video)  # warm-up
torch.npu.synchronize()
prof = torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(OUT),
)
prof.start()
vae.encode(video)
torch.npu.synchronize()
prof.stop()
print("done", OUT)
