"""Expose the torch-only ``transformer_engine`` stand-in (``npu_shims/``) without fooling megatron-core.

The Cosmos DiT imports ``transformer_engine`` at module level; on Ascend the real package
cannot be installed, so ``npu_shims/transformer_engine`` provides the few pieces the DiT uses.
megatron-core (imported by the Cosmos VAE) decides whether TE exists by catching
``ModuleNotFoundError`` and would take its TE code paths if it saw the shim.  So: if the real
package is importable, do nothing; otherwise import megatron-core first (it records "no TE"),
then put the shim on ``sys.path``.  Call before importing ``cosmos_predict2`` networks.
"""

import sys
from pathlib import Path

SHIM_DIR = Path(__file__).resolve().parents[4] / "npu_shims"


def ensure_transformer_engine() -> str:
    """Returns ``"real"`` or ``"shim"``."""
    try:
        import transformer_engine

        return "shim" if getattr(transformer_engine, "IS_TORCH_SHIM", False) else "real"
    except ModuleNotFoundError:
        pass
    try:
        import megatron.core  # noqa: F401  (resolves HAVE_TE=False before the shim is visible)
    except ImportError:
        pass
    if str(SHIM_DIR) not in sys.path:
        sys.path.insert(0, str(SHIM_DIR))
    import transformer_engine  # noqa: F401

    return "shim"


# ----------------------------------------------------------------------------- fp32 strategy
# Upstream 2B training runs with ``use_wan_fp32_strategy=True``: the time embedding, every
# block's AdaLN modulation and the final layer run under ``autocast(float32)``, and q / k are
# cast to float32 before RoPE.  Ascend autocast supports only fp16 / bf16, so the autocast is a
# no-op there (and the final layer's float32 assertion fails).  Instead these small modules are
# made float32 "islands": inputs are cast to float32 and their Linear layers compute with
# float32 copies of their weights — the same arithmetic as autocast(float32), independent of the
# parameter storage dtype (bf16 under DeepSpeed).  Block outputs are cast back by the existing
# ``type_as`` calls; the final layer's float32 output is cast back by ``finalize_block_loop``.

FP32_ISLANDS = ("t_embedder", "t_embedding_norm", "final_layer",
                "adaln_modulation_self_attn", "adaln_modulation_cross_attn", "adaln_modulation_mlp")


def _cast_inputs_fp32(_module, args, kwargs):
    import torch

    def cast(value):
        return value.float() if torch.is_tensor(value) and value.is_floating_point() else value

    return tuple(cast(a) for a in args), {k: cast(v) for k, v in kwargs.items()}


def _fp32_linear(self, x):
    import torch.nn.functional as F

    return F.linear(x.float(), self.weight.float(), None if self.bias is None else self.bias.float())


def apply_fp32_strategy(net) -> int:
    """Make the time embedding, AdaLN modulation and final layer float32 islands; return their count."""
    import types

    from torch import nn

    islands = 0
    for name, module in net.named_modules():
        if name.rsplit(".", 1)[-1] in FP32_ISLANDS and not getattr(module, "_fp32_island", False):
            module.register_forward_pre_hook(_cast_inputs_fp32, with_kwargs=True)
            for sub in module.modules():
                if isinstance(sub, nn.Linear):
                    sub.forward = types.MethodType(_fp32_linear, sub)
            module._fp32_island = True
            islands += 1
        if hasattr(module, "is_selfattn") and hasattr(module, "use_wan_fp32_strategy"):
            module.use_wan_fp32_strategy = True       # q / k in float32 before RoPE (explicit casts, no autocast)
    return islands
