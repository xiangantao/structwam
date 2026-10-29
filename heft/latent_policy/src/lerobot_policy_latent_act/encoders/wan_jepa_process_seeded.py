"""Per-episode (per-clip) feature seeds for the direct Wan+V-JEPA2 arm.

Identical to ``wan_jepa_process`` except that the request's seed argument is derived per
sample from ``episode_index`` instead of being one constant for the whole batch.  The
motivation is measured: with one constant seed the noise field is bit-identical for every
clip, so the policy can absorb it; with a per-episode seed that shared artifact disappears
while the within-clip-invariant content is untouched.

Enable with ``policy.feature_seed_mode = "per_clip"``; the default (``"fixed"``) path is
untouched.  The worker script is selected with ``LATENT_WAN_JEPA_WORKER`` because the base
adapter derives it from its own module path.
"""

from __future__ import annotations

import torch

from .wan_jepa_process import WanJepaProcessEncoder


class WanJepaProcessEncoderSeeded(WanJepaProcessEncoder):
    """Same transport as the base class; only the seed argument changes."""

    @staticmethod
    def clip_seeds(episode_indices, base_seed: int) -> list[int]:
        # Distillation uses sample_seed(42, "distillation-teacher", domain, row_id); the
        # episode index plays the role of row_id here, and the tag is deliberately
        # different so this arm never shares a noise field with the distillation teacher.
        return [int(base_seed) + 1_000_003 * int(episode) for episode in episode_indices]

    @torch.no_grad()
    def encode(self, images, episode_indices=None):
        if episode_indices is None:
            raise ValueError(
                "per_clip feature seeds require episode_indices; refusing to fall back to a "
                "constant seed, which would silently invalidate the comparison"
            )
        count = len(episode_indices)
        if count != images[0].shape[0]:
            raise ValueError(
                f"episode_indices has {count} entries for a batch of {images[0].shape[0]}"
            )
        # The worker flattens every camera into the clip list, so each camera needs its own
        # copy of the batch's seeds: order is camera-major, then batch order within a camera.
        per_camera = self.clip_seeds(episode_indices, self.config.feature_seed)
        seeds = per_camera * len(images)
        return super().encode(images, seeds=seeds)
