"""ACT defaults plus a frozen video encoder and modality-specific history."""

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.constants import OBS_ENV_STATE, OBS_STATE

from .data import SUITES, canonical_task


@PreTrainedConfig.register_subclass("latent_act")
@dataclass
class LatentACTConfig(ACTConfig):
    encoder: str = "wan"
    suite: str = "libero_spatial"
    task_names: list[str] = field(default_factory=list)
    history_frames: int = 16
    image_size: int = 256
    camera_keys: list[str] = field(
        default_factory=lambda: ["observation.images.image", "observation.images.image2"]
    )
    chunk_size: int = 16
    n_action_steps: int = 8
    pretrained_backbone_weights: str | None = None
    push_to_hub: bool = False
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ENV": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )
    wan_model_path: str = ""
    wan_runtime_python: str = ""
    channel_mask: str = ""
    channel_mask_sha256: str = ""
    vjepa_root: str = ""
    vjepa_checkpoint: str = ""
    vjepa_checkpoint_sha256: str = ""
    jepa_model_path: str = ""
    artifact_dir: str = ""
    distilled_root: str = ""
    distilled_checkpoint: str = ""
    distilled_checkpoint_sha256: str = ""
    distilled_runtime_python: str = ""
    extract_batch_size: int = 8
    extract_lanes: int = 1
    compile_blocks: bool = False
    feature_seed: int = 42
    # "fixed" (original) sends one seed for the whole batch, so the request noise field is
    # bit-identical across clips; "per_clip" derives one seed per episode instead.
    feature_seed_mode: str = "fixed"
    need_attention_weights: bool = False
    checkpoint_encoder: bool = True

    def __post_init__(self):
        # ACT keeps one current state. Image history uses the existing separate dataset property.
        super().__post_init__()
        if self.encoder not in ("wan", "vjepa2", "wan_jepa", "distilled"):
            raise ValueError("encoder must be wan, vjepa2, or distilled")
        if self.suite not in SUITES:
            raise ValueError(f"Unknown LIBERO suite: {self.suite}")
        if self.history_frames != 16 or self.image_size != 256:
            raise ValueError("This comparison fixes 16 input frames at 256x256")
        if len(self.camera_keys) != 2 or len(set(self.camera_keys)) != 2:
            raise ValueError("Exactly two distinct camera keys are required")
        if self.dim_model % 4:
            raise ValueError("dim_model must be divisible by 4 for ACT spatial positions")
        if min(self.chunk_size, self.n_action_steps, self.extract_batch_size) < 1:
            raise ValueError("Chunk, action and extraction batch sizes must be positive")
        if self.extract_lanes not in (1, 2):
            raise ValueError("extract_lanes must be 1 or 2")
        if self.encoder == "vjepa2" and (self.compile_blocks or self.extract_lanes != 1):
            raise ValueError("Wan extraction lane/compile options do not apply to V-JEPA2")
        if self.encoder in ("distilled", "wan_jepa") and (self.compile_blocks or self.extract_lanes != 1):
            raise ValueError("Wan extraction lane/compile options do not apply to distilled")
        if self.task_names and (
            len(self.task_names) != 10 or len({canonical_task(t) for t in self.task_names}) != 10
        ):
            raise ValueError("task_names must contain the suite's 10 distinct descriptions")
        if self.normalization_mapping.get("VISUAL") != NormalizationMode.IDENTITY:
            raise ValueError(
                "Images must remain in [0,1]; normalization belongs to the frozen encoder"
            )
        if self.normalization_mapping.get("ENV") != NormalizationMode.IDENTITY:
            raise ValueError("Task one-hot must not be normalized")

    @property
    def visual_dim(self) -> int:
        return {"wan": 896, "vjepa2": 1024, "wan_jepa": 1152, "distilled": 1152}[self.encoder]

    @property
    def visual_shape(self) -> tuple[int, int, int, int]:
        """Return the frozen encoder's ``(frames, channels, height, width)`` contract."""
        return {
            "wan": (16, 896, 14, 14),
            "vjepa2": (8, 1024, 16, 16),
            "wan_jepa": (8, 1152, 16, 16),
            "distilled": (8, 1152, 16, 16),
        }[self.encoder]

    @property
    def image_observation_delta_indices(self) -> list[int]:
        return list(range(1 - self.history_frames, 1))

    @property
    def state_observation_delta_indices(self) -> None:
        return None

    def validate_features(self) -> None:
        super().validate_features()
        if set(self.image_features) != set(self.camera_keys):
            raise ValueError(
                f"Expected cameras {self.camera_keys}, got {list(self.image_features)}"
            )
        if len(self.task_names) != 10:
            raise ValueError(
                "Prepare a suite-specific config with all 10 task_names before training"
            )
        if self.input_features[OBS_STATE].shape != (8,) or self.action_feature.shape != (7,):
            raise ValueError("LIBERO policy requires an 8-D state and 7-D action")
        for feature in self.image_features.values():
            if tuple(feature.shape) != (3, self.image_size, self.image_size):
                raise ValueError("LIBERO camera features must be RGB 256x256")
        expected = PolicyFeature(type=FeatureType.ENV, shape=(10,))
        previous = self.input_features.get(OBS_ENV_STATE)
        if previous is not None and previous != expected:
            raise ValueError("environment_state is reserved for the 10-D task one-hot")
        self.input_features[OBS_ENV_STATE] = expected
        self.input_features = {
            **{
                key: value
                for key, value in self.input_features.items()
                if key not in self.camera_keys
            },
            **{key: self.input_features[key] for key in self.camera_keys},
        }
