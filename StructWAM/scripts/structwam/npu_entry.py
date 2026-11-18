#!/usr/bin/env python3
"""Run an OpenWAM entry script on Ascend NPUs: install torch_npu's CUDA shim first.

    torchrun ... scripts/structwam/npu_entry.py scripts/train.py --config-name train_struct ...
"""

import runpy
import sys

import torch_npu  # noqa: F401
from torch_npu.contrib import transfer_to_npu  # noqa: F401

# Opt-in (default unchanged): OPENWAM_NPU_JIT_COMPILE=0 uses the precompiled binary kernels instead of compiling
# operators online, which recompiles whenever an input shape changes.
import os

if os.environ.get("OPENWAM_NPU_JIT_COMPILE", "").strip() == "0":
    torch_npu.npu.set_compile_mode(jit_compile=False)

if __name__ == "__main__":
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
