"""Frozen Wan+V-JEPA2 student used as a fused visual feature encoder.

The student checkpoint is loaded only in the optional extraction runtime. It
is deliberately kept outside the ACT module tree so the student and its
mapping/heads cannot enter the ACT optimizer or policy checkpoint.
"""

from pathlib import Path

import torch

from .common import checked_local_file, rgb_uint8


class DistilledEncoder:
    """Run the locally trained student and expose only its fused feature map."""

    def __init__(self, config, device):
        self.config = config
        self.device = torch.device(device)
        self.extract_batch_size = int(getattr(config, "extract_batch_size", 1))
        if self.extract_batch_size < 1:
            raise ValueError("extract_batch_size must be positive")

        root = Path(config.distilled_root).resolve(strict=True)
        checkpoint = checked_local_file(
            config.distilled_checkpoint, config.distilled_checkpoint_sha256
        )
        vjepa_root = Path(config.vjepa_root).resolve(strict=True)
        if not (root / "latent_video/distillation/model.py").is_file():
            raise ValueError(f"Not a distillation source tree: {root}")
        if not (vjepa_root / "src/models/vision_transformer.py").is_file():
            raise ValueError(f"Not a V-JEPA2 source tree: {vjepa_root}")

        # Import only the student model. The public inference helper imports
        # distillation.train, which imports the frozen Wan teacher and its
        # modified Diffusers stack even though inference needs neither. That
        # would reintroduce the Transformers-version conflict this worker is
        # intended to isolate.
        import importlib
        import sys
        from collections.abc import Mapping

        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        model_module = importlib.import_module("latent_video.distillation.model")
        module_path = Path(model_module.__file__).resolve()
        if not module_path.is_relative_to(root):
            raise RuntimeError(
                f"Distillation import resolved outside configured root: {module_path}"
            )
        architecture = {
            "backbone": "vjepa2_vit_large",
            "width": 1024,
            "grid": [8, 16, 16],
            "mapping_depth": 2,
            "attention_heads": 16,
            "ffn_width": 4096,
            "mapping_rope": "THW-adjacent-pairs20/22/22-base10000",
            "residual_scale_init": 1e-3,
            "projection": "Linear+Linear-GELU-Linear",
            "projection_hidden": 1024,
            "groups": [list(group) for group in model_module.GROUPS],
            "backbone_trainable": True,
        }
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != "heft.wan_jepa.distillation.v1"
            or payload.get("architecture") != architecture
            or not isinstance(payload.get("student"), Mapping)
        ):
            raise ValueError("Incompatible distilled student checkpoint")
        if str(vjepa_root) in sys.path:
            sys.path.remove(str(vjepa_root))
        sys.path.insert(0, str(vjepa_root))
        vit = importlib.import_module("src.models.vision_transformer")
        vit_path = Path(vit.__file__).resolve()
        if not vit_path.is_relative_to(vjepa_root):
            raise RuntimeError(f"V-JEPA import resolved outside configured root: {vit_path}")
        backbone = vit.vit_large(
            img_size=256,
            num_frames=16,
            patch_size=16,
            tubelet_size=2,
            uniform_power=True,
            use_rope=True,
            use_activation_checkpointing=False,
        )
        self.model = model_module.VideoStudent(
            backbone, activation_checkpointing=False
        )
        self.model.load_state_dict(payload["student"], strict=True)
        del payload
        self.model.to(self.device).eval().requires_grad_(False)
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("Distilled student must be frozen and in eval mode")

    @torch.no_grad()
    def encode(self, images: list[torch.Tensor]) -> list[torch.Tensor]:
        """Return one fused map per camera, preserving camera and batch order.

        Inputs use the shared ACT contract ``[B,16,3,256,256]``. The student
        performs its own ImageNet normalization after changing the layout to
        ``[B,3,16,256,256]``. No temporal clip is selected here: every input
        row is encoded exactly once, in chunks controlled by the config.
        """

        if not images:
            raise ValueError("At least one camera is required")
        images = rgb_uint8(images, frames=16, size=256)
        batch_size = images[0].shape[0]
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        clips = [clip for camera in images for clip in camera]
        output = []
        self.model.eval()
        for start in range(0, len(clips), self.extract_batch_size):
            group = torch.stack(clips[start : start + self.extract_batch_size])
            group = group.to(self.device, non_blocking=self.device.type == "cuda")
            with torch.autocast(
                self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                feature_map = self.model.feature_map(group)
            expected = (len(group), 1152, 8, 16, 16)
            if tuple(feature_map.shape) != expected:
                raise ValueError(
                    f"Unexpected distilled student output: {tuple(feature_map.shape)}; "
                    f"expected {expected}"
                )
            # feature_map is [B,C,T,H,W], while ACT consumes [B,T,C,H,W].
            output.append(feature_map.permute(0, 2, 1, 3, 4).contiguous())

        # Keep the result a normal tensor after no_grad, since ACT's trainable
        # 1x1 projection must be able to save it for its own backward pass.
        with torch.inference_mode(False):
            features = torch.cat(output, dim=0).detach().clone()
        return list(features.split(batch_size))

    def health(self):
        """Return invariants used to validate worker startup."""

        return {
            "training": bool(self.model.training),
            "any_requires_grad": any(p.requires_grad for p in self.model.parameters()),
            "output_shape": [8, 1152, 16, 16],
        }

    def close(self):
        # Dropping the reference lets the worker release CUDA allocations before
        # the process exits.
        self.model = None
