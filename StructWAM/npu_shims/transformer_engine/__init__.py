"""Torch-only stand-in for NVIDIA transformer-engine, for running the Cosmos-Predict2.5 DiT on
Ascend NPUs (transformer-engine needs CUDA / nvcc and cannot be installed there).

Only what the Cosmos DiT touches is provided:

* ``pytorch.RMSNorm``                      — q / k / time-embedding norms (same parameter names,
  plus an ``_extra_state`` slot so TE checkpoints load without missing / unexpected keys);
* ``pytorch.attention.apply_rotary_pos_emb`` (also ``pytorch.attention.rope``) — TE's RoPE
  (non-interleaved rotate-half, ``sbhd`` / ``bshd``), computed in float32 like the fused kernel;
* ``pytorch.distributed`` / ``pytorch.module.base`` — import-time names used by Cosmos utilities.

``DotProductAttention`` is deliberately not provided: build the DiT with ``atten_backend=torch``
(``video_backbone.atten_backend`` in the OpenWAM config).  Do not put ``npu_shims`` on
``PYTHONPATH``: megatron-core would mistake the shim for the real package.  The Cosmos builder
exposes it via ``openwam/model/video_backbone/cosmos_predict25/npu_compat.py``.
"""

from transformer_engine import pytorch  # noqa: F401

__version__ = "0.0.0+torch-shim"
IS_TORCH_SHIM = True
