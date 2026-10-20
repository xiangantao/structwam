"""``transformer_engine.pytorch.attention`` stand-in."""

from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb


class DotProductAttention:  # pragma: no cover - guard only
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "transformer-engine DotProductAttention is not available on this host (torch shim). "
            "Build the Cosmos DiT with atten_backend='torch' (video_backbone.atten_backend)."
        )


__all__ = ["DotProductAttention", "apply_rotary_pos_emb"]
