"""Reuse the existing five-candidate extractor, including its fixed noise and channel choices."""

import torch

from .common import checked_local_file, rgb_uint8


class WanEncoder:
    def __init__(self, config, device):
        from latent_video.classification.execution import ExtractionPool
        from latent_video.config import ClipConfig
        from latent_video.extractor import WanLatentExtractor

        self.config = config
        self.device = torch.device(device)
        mask = checked_local_file(config.channel_mask, config.channel_mask_sha256)
        self.extractor = WanLatentExtractor.from_local(
            config.wan_model_path,
            channel_mask=mask,
            held_out=None,
            mask_key="ablation_iterative_256",
            device=self.device,
            config=ClipConfig(frames=16, resolution=(256, 256), grid=(14, 14)),
        )
        extractors = [self.extractor]
        if config.extract_lanes == 2:
            extractors.append(self.extractor.replica())
        self.pool = ExtractionPool(extractors, compile_blocks=config.compile_blocks)

    @torch.no_grad()
    def encode(self, images: list[torch.Tensor]) -> list[torch.Tensor]:
        from latent_video.classification.encoder import fuse_candidates

        images = rgb_uint8(images, self.config.history_frames, self.config.image_size)
        batch_size = images[0].shape[0]
        clips = [clip for camera in images for clip in camera]
        requests = []
        for start in range(0, len(clips), self.config.extract_batch_size):
            group = clips[start : start + self.config.extract_batch_size]
            requests.append(
                {
                    "frames": group,
                    "seeds": [self.config.feature_seed] * len(group),
                    "output_device": self.device,
                }
            )
        with torch.autocast(self.device.type, enabled=False):
            groups = self.pool.extract_groups(requests)
        fused = []
        for request, group in zip(requests, groups, strict=True):
            if len(group) != len(request["frames"]):
                raise ValueError("Extractor returned the wrong clip count")
            fused.extend(fuse_candidates(result, self.extractor) for result in group)
        # Inference tensors cannot be saved by a trainable Conv2d's backward.
        with torch.inference_mode(False):
            outputs = torch.stack(fused).detach().clone()
        return list(outputs.split(batch_size))

    def close(self):
        self.pool.close()
