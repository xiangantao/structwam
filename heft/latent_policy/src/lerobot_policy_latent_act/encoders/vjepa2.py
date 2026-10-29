"""Official V-JEPA2 ViT-L with strict, local-only weight loading."""

import importlib
import sys
from pathlib import Path

import torch

from .common import checked_local_file, rgb_uint8


def load_backbone_state(model, checkpoint: dict, key: str = "target_encoder"):
    state = checkpoint[key]
    cleaned = {}
    for name, tensor in state.items():
        while name.startswith(("module.", "backbone.")):
            name = name.split(".", 1)[1]
        if name in cleaned:
            raise ValueError(f"Duplicate checkpoint key after prefix removal: {name}")
        cleaned[name] = tensor
    # Unlike the classification loader, never substitute random tensors on mismatch.
    model.load_state_dict(cleaned, strict=True)


class VJepa2Encoder:
    def __init__(self, config, device):
        self.config = config
        self.device = torch.device(device)
        root = Path(config.vjepa_root).resolve(strict=True)
        checkpoint = checked_local_file(config.vjepa_checkpoint, config.vjepa_checkpoint_sha256)
        if not (root / "src/models/vision_transformer.py").is_file():
            raise ValueError(f"Not a V-JEPA2 source tree: {root}")
        sys.path.insert(0, str(root))
        try:
            vit = importlib.import_module("src.models.vision_transformer")
        finally:
            sys.path.remove(str(root))
        if Path(vit.__file__).resolve() != root / "src/models/vision_transformer.py":
            raise RuntimeError("Another 'src' package shadows the configured V-JEPA2 repository")
        self.model = vit.vit_large(
            img_size=config.image_size,
            num_frames=config.history_frames,
            patch_size=16,
            tubelet_size=2,
            uniform_power=True,
            use_rope=True,
            img_temporal_dim_size=None,
        )
        load_backbone_state(
            self.model, torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        self.model.requires_grad_(False).eval().to(self.device)

    @torch.no_grad()
    def encode(self, images: list[torch.Tensor]) -> list[torch.Tensor]:
        images = rgb_uint8(images, self.config.history_frames, self.config.image_size)
        count = images[0].shape[0]
        clips = torch.cat(images)
        output = []
        self.model.eval()
        for group in clips.split(self.config.extract_batch_size):
            x = group.to(self.device, dtype=torch.float32).div_(255)
            mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
            std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
            x = ((x - mean) / std).permute(0, 2, 1, 3, 4).contiguous()
            with torch.autocast(
                self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
            ):
                tokens = self.model(x)
            expected = (len(group), 8 * 16 * 16, 1024)
            if tuple(tokens.shape) != expected:
                raise ValueError(f"Unexpected V-JEPA2 output: {tokens.shape}; expected {expected}")
            output.append(tokens.reshape(len(group), 8, 16, 16, 1024).permute(0, 1, 4, 2, 3))
        with torch.inference_mode(False):
            features = torch.cat(output).detach().clone()
        return list(features.split(count))

    def close(self):
        pass
