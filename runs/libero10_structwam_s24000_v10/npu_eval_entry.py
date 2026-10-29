"""scripts/structwam/npu_entry.py plus one deploy-only fix, applied here so the repo stays untouched
while training runs from the same SFS checkout.

PhaseGate.forward reshapes proprio by its own first dim; the deploy engine passes a 1-D [state_dim]
tensor, which became [state_dim, 1] and failed in the gate's Linear.  Reshape by the pooled batch
instead (identical for the [B, state_dim] training inputs).
"""

import runpy
import sys

import torch_npu  # noqa: F401
from torch_npu.contrib import transfer_to_npu  # noqa: F401

from openwam.structure import phase

_gate_forward = phase.PhaseGate.forward


def _forward(self, pooled, proprio=None):
    if proprio is not None:
        proprio = proprio.reshape(pooled.shape[0], -1)
    return _gate_forward(self, pooled, proprio)


phase.PhaseGate.forward = _forward

if __name__ == "__main__":
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
