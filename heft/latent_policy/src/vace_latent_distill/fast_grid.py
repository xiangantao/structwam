"""Faster ``GridBlock`` forward for Ascend inference (decoders and heads of the student).

* Shifted / plain window partition and its inverse become one ``index_select`` each over
  the flattened grid (cached per grid and shift) instead of ``torch.roll`` plus
  reshape/permute; the shifted-window mask is cached per batch shape.  Pure data movement:
  the same values as the original.
* The temporal attention (sequences of ~13-21 steps, hundreds of thousands of them) uses an
  explicit batched matmul + softmax; the fused SDPA kernel is about twice slower at that
  shape.  This changes bf16 rounding (max abs difference ~0.02 on attention outputs), so
  features computed with and without the patch are close but not bit-identical; use the
  same path for training and rollout.
"""

import types

import torch


def _orders(block, height, width, device):
    key = (height, width, device)
    if key not in block._fast_orders:
        w, s = block.window, block.shift
        ys, xs = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        source = ((ys + s) % height) * width + (xs + s) % width
        order = source.reshape(height // w, w, width // w, w).permute(0, 2, 1, 3).reshape(-1).to(device)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(len(order), device=device)
        block._fast_orders[key] = (order, inverse)
    return block._fast_orders[key]


def _allowed(block, batch, temporal, height, width, device):
    key = (batch, temporal, height, width, device)
    if key not in block._fast_masks:
        y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        y = (y + block.shift) % height
        x = (x + block.shift) % width
        labels = (y >= block.shift).long() * 2 + (x >= block.shift).long()
        labels = block.windows(labels[None, None, :, :, None]).squeeze(-1)
        block._fast_masks = {key: (labels[:, :, None] == labels[:, None, :]).repeat(batch * temporal, 1, 1)}
    return block._fast_masks[key]


def _short_attention(attention, query, memory=None):
    """``Attention.forward`` for short sequences without masks, via explicit matmuls."""
    memory = query if memory is None else memory
    batch, count, dim = query.shape
    heads = attention.heads
    q = attention.q(query).reshape(batch, count, heads, dim // heads).transpose(1, 2)
    k, v = attention.kv(memory).reshape(batch, -1, 2, heads, dim // heads).permute(2, 0, 3, 1, 4).unbind(0)
    scores = (q @ k.transpose(-2, -1)) * (dim // heads) ** -0.5
    result = scores.softmax(-1) @ v
    return attention.proj(result.transpose(1, 2).reshape(batch, count, dim))


def _forward(self, field, history=None):
    batch, temporal, height, width, dim = field.shape
    order, inverse = _orders(self, height, width, field.device)
    value = self.space_norm(field).reshape(batch, temporal, height * width, dim)
    windows = value.index_select(2, order).reshape(-1, self.window * self.window, dim)
    allowed = _allowed(self, batch, temporal, height, width, field.device) if self.shift else None
    value = self.space_attention(windows, allowed=allowed)
    value = value.reshape(batch, temporal, height * width, dim).index_select(2, inverse)
    field = field + value.reshape(batch, temporal, height, width, dim)
    value = self.time_norm(field).permute(0, 2, 3, 1, 4).reshape(-1, temporal, dim)
    context = None
    if history is not None:
        past = self.time_norm(history).permute(0, 2, 3, 1, 4).reshape(-1, history.shape[1], dim)
        context = torch.cat((past, value), 1)
    value = (
        _short_attention(self.time_attention, value, context)
        .reshape(batch, height, width, temporal, dim)
        .permute(0, 3, 1, 2, 4)
    )
    field = field + value
    return field + self.ffn(self.ffn_norm(field))


def install(model):
    """Patch every ``GridBlock`` of a student in place; returns the count."""
    patched = 0
    for module in model.modules():
        if type(module).__name__ == "GridBlock" and not hasattr(module, "_fast_orders"):
            module._fast_orders, module._fast_masks = {}, {}
            module.forward = types.MethodType(_forward, module)
            patched += 1
    return patched
