"""Cached, single-pass 3D RoPE for the V-JEPA2 backbone (same values as the original).

V-JEPA2's ``RoPEAttention`` rebuilds sin/cos for every forward and rotates the depth,
height and width slices of q and k separately (six strided element-wise passes plus a
concatenation).  On Ascend that costs about twice the attention itself.  Here the three
per-axis tables are built once per (positions, dtype, device) with the original
``rotate_queries_or_keys`` arithmetic, laid side by side over the head dimension (the
unrotated tail gets cos 1 / sin 0), and q and k are rotated in one pass each.  Pairs are
adjacent channels in both versions and every slice boundary is even, so each element sees
the same cos/sin and the same multiply-add as before.

``install(..., half_split=True)`` additionally reorders each head's q/k channels (folded
into the ``qkv`` weights once) so the two members of every rotation pair sit in the first
and second half of the head: the rotation then needs only contiguous slices instead of an
interleaved unbind/stack, which is slow on Ascend.  Every rotated element still gets the
same operands and multiply-add; q.k is unchanged mathematically (the dot product sums the
same products in another order), so attention matches up to bf16 summation order.
"""

import types

import torch
from torch.nn import functional as F


def _tables(attention, count, height, width, dtype, device):
    ids = torch.arange(count, device=device)
    depth_ids, height_ids, width_ids = attention.separate_positions(ids, height, width)
    cos, sin = [], []
    for positions, dim in ((depth_ids, attention.d_dim), (height_ids, attention.h_dim), (width_ids, attention.w_dim)):
        # Same arithmetic as rotate_queries_or_keys: omega in the activation dtype, duplicated halves.
        omega = torch.arange(dim // 2, dtype=dtype, device=device)
        omega /= dim / 2.0
        omega = 1.0 / 10000**omega
        freq = torch.einsum("..., f -> ... f", positions, omega)
        cos.append(freq.cos().squeeze(-1).repeat(1, 1, 1, 2))
        sin.append(freq.sin().squeeze(-1).repeat(1, 1, 1, 2))
    rest = attention.head_dim - attention.d_dim - attention.h_dim - attention.w_dim
    if rest:
        cos.append(torch.ones(1, 1, count, rest, dtype=cos[0].dtype, device=device))
        sin.append(torch.zeros(1, 1, count, rest, dtype=sin[0].dtype, device=device))
    return torch.cat(cos, -1), torch.cat(sin, -1)


def _rotate(x, cos, sin):
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    return x * cos + torch.stack((-y2, y1), dim=-1).flatten(-2) * sin


def _rotate_half(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), -1) * sin


def _pair_order(head_dim):
    """Half-split position i -> original channel: pair p's members 2p, 2p+1 go to p, p + D/2."""
    half = head_dim // 2
    return torch.cat((torch.arange(half) * 2, torch.arange(half) * 2 + 1))


def _forward(self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
    if mask is not None or T is None or H_patches is None or W_patches is None:
        if self._half_split:
            raise RuntimeError("half-split q/k weights need the cached-RoPE path (no token mask, T/H/W given)")
        return self._original_forward(x, mask=mask, attn_mask=attn_mask, T=T, H_patches=H_patches, W_patches=W_patches)
    batch, count, channels = x.size()
    qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    key = (count, T, H_patches, W_patches, q.dtype, q.device)
    if key not in self._rope_cache:
        cos, sin = _tables(self, count, H_patches, W_patches, q.dtype, q.device)
        if self._half_split:
            order = _pair_order(self.head_dim).to(q.device)
            cos, sin = cos[..., order], sin[..., order]
        self._rope_cache[key] = (cos, sin)
    cos, sin = self._rope_cache[key]
    rotate = _rotate_half if self._half_split else _rotate
    q, k = rotate(q, cos, sin), rotate(k, cos, sin)
    x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.proj_drop_prob if self.training else 0.0,
                                       is_causal=self.is_causal, attn_mask=attn_mask)
    x = x.transpose(1, 2).reshape(batch, count, channels)
    return self.proj_drop(self.proj(x))


@torch.no_grad()
def _reorder_qk(module):
    """Permute each head's q and k output channels of ``qkv`` into half-split pair order."""
    order = _pair_order(module.head_dim).to(module.qkv.weight.device)
    heads, head_dim = module.num_heads, module.head_dim
    weight = module.qkv.weight.view(3, heads, head_dim, -1)
    weight[:2] = weight[:2, :, order]
    if module.qkv.bias is not None:
        bias = module.qkv.bias.view(3, heads, head_dim)
        bias[:2] = bias[:2, :, order]


def install(backbone, half_split=False):
    """Patch every RoPE attention module of a V-JEPA2 backbone in place; returns the count.

    ``half_split`` reorders the q/k weights (in place, inference only) for the faster rotation;
    the module's outputs do not change except for bf16 summation order inside attention.
    """
    patched = 0
    for module in backbone.modules():
        if type(module).__name__ == "RoPEAttention" and not hasattr(module, "_rope_cache"):
            module._original_forward = module.forward
            module._rope_cache = {}
            module._half_split = half_split
            if half_split:
                _reorder_qk(module)
            module.forward = types.MethodType(_forward, module)
            patched += 1
    return patched
