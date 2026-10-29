"""Independent configuration for the SSv2 ViT-L baseline."""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from ...config import (
    HEFT_ROOT,
    DataConfig,
    OptimizationConfig,
    WandbConfig,
    positive_int,
)


@dataclass(frozen=True)
class ClipConfig:
    frames: int = 16

    def __post_init__(self):
        if type(self.frames) is not int or self.frames != 16:
            raise ValueError("The comparison requires 16 frames per segment")


@dataclass(frozen=True)
class RunConfig:
    data: DataConfig
    model_path: str = "../models/vjepa2/vitl.pt"
    vjepa_root: str = "../vjepa2"
    output_dir: str = "../runs/ssv2_vjepa2_vitl_singlehead"
    seed: int = 42
    num_workers: int = 4
    extract_batch_size: int = 32
    extract_lanes: int = 1
    num_heads: int = 16
    num_probe_blocks: int = 4
    validate_every: int = 0
    clip: ClipConfig = field(default_factory=ClipConfig)
    optimization: OptimizationConfig = field(
        default_factory=lambda: OptimizationConfig(batch_size=32)
    )
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if type(self.num_workers) is not int or self.num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")
        for name in ("extract_batch_size", "num_heads", "num_probe_blocks"):
            positive_int(name, getattr(self, name))
        if type(self.extract_lanes) is not int or self.extract_lanes not in (1, 2):
            raise ValueError("extract_lanes must be 1 or 2")
        if self.num_heads != 16 or self.num_probe_blocks != 4:
            raise ValueError("The comparison uses one depth-4, 16-attention-head probe")
        if self.data.crop_size != 256 or self.data.num_segments != 2:
            raise ValueError("ViT-L comparison requires 256px and two segments")
        if self.data.frame_step != 4 or self.data.num_views_per_segment != 3:
            raise ValueError("Require frame_step=4 and three evaluation views")
        if self.extract_batch_size < self.data.num_segments:
            raise ValueError("extract_batch_size must accommodate one video's segments")
        if type(self.validate_every) is not int or self.validate_every != 0:
            raise ValueError(
                "Use separate checkpoint evaluation; validate_every must be zero"
            )

    def path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else HEFT_ROOT / path).resolve()

    def metadata(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        with Path(path).open() as stream:
            value = yaml.safe_load(stream)
        if not isinstance(value, dict):
            raise TypeError("Configuration must be a YAML mapping")
        for name, factory in (
            ("data", DataConfig),
            ("clip", ClipConfig),
            ("optimization", OptimizationConfig),
            ("wandb", WandbConfig),
        ):
            if name in value:
                if not isinstance(value[name], dict):
                    raise TypeError(f"{name} must be a mapping")
                value[name] = factory(**value[name])
        return cls(**value)
