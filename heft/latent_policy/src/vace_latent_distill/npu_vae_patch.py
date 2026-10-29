"""Optional faster CausalConv3d.forward for the native Wan VAE (Ascend NPU).

The reference forward does ``cat(cache, x)`` then ``F.pad`` on all of (W, H, T) and
then a zero-padding conv.  Here the spatial padding is handed to the convolution
itself and ``F.pad`` only runs when a temporal front pad is still needed, so the
common cached-chunk case launches no pad kernel.  VEGA-3D is not modified: the
class is patched at runtime, and ``set_enabled(False)`` restores the reference path.
"""
import os

import torch
from torch.nn import functional as F

_STATE = {"on": False, "installed": False}


def _reference_forward(self, x, cache_x=None):
    padding = list(self._padding)
    if cache_x is not None and self._padding[4] > 0:
        cache_x = cache_x.to(x.device)
        x = torch.cat([cache_x, x], dim=2)
        padding[4] -= cache_x.shape[2]
    x = F.pad(x, padding)
    return torch.nn.Conv3d.forward(self, x)


def _fast_forward(self, x, cache_x=None):
    if not _STATE["on"]:
        return _reference_forward(self, x, cache_x)
    t_pad = self._padding[4]
    joined = x
    if cache_x is not None and t_pad > 0:
        joined = torch.cat([cache_x.to(x.device), x], dim=2)
        t_pad -= cache_x.shape[2]
    if t_pad < 0:  # reference would crop frames with a negative pad; keep that path as is
        return _reference_forward(self, x, cache_x)
    if t_pad > 0:
        joined = F.pad(joined, (0, 0, 0, 0, t_pad, 0))
    return F.conv3d(joined, self.weight, self.bias, self.stride,
                    (0, self._padding[2], self._padding[0]), self.dilation, self.groups)


def set_enabled(flag: bool):
    _STATE["on"] = bool(flag)


def install(vae, enabled=None):
    """Patch CausalConv3d of ``vae`` (a WanVAE wrapper). Idempotent; env VACE_VAE_OPT=0 disables."""
    if enabled is None:
        enabled = os.environ.get("VACE_VAE_OPT", "1") == "1"
    if not _STATE["installed"]:
        for module in vae.model.modules():
            if type(module).__name__ == "CausalConv3d":
                type(module).forward = _fast_forward
                _STATE["installed"] = True
                break
    set_enabled(enabled)
