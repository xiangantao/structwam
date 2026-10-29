"""Explicit input boundaries and the two native teacher timelines."""

import dataclasses
import json
from pathlib import Path

HEAD_TASKS = ("seg_l22h9", "seg_l15h7", "tracking", "semantic", "depth")
TASKS = (*HEAD_TASKS, "shared")
OBSERVED_FRAMES = 33
FUTURE_FRAMES = 16
WARMUP_FRAMES = 15
OBSERVED_ANCHORS = (4, 6, 8, 10)
FUTURE_ANCHORS = (8, 10)
NEGATIVE_PROMPT = "overexposed, blurry, subtitles, low quality, deformed, extra fingers, camera cut"


@dataclasses.dataclass
class ModelConfig:
    task: str = "shared"
    vjepa_root: str = "../vjepa2"
    vjepa_weights: str = "../models/vjepa2/vitl.pt"
    image_size: int = 256
    grid_size: int = 32
    dim: int = 1024
    text_dim: int = 4096
    channels: int = 128
    attention_heads: int = 16
    language_layers: int = 4
    decoder_layers: int = 2
    head_layers: int = 2
    window_size: int = 8
    mlp_ratio: int = 2
    activation_checkpointing: bool = True
    # "shared": one future query added to every copy of the last observed frame;
    # "per_time": one learned query per future readout slot.
    future_query: str = "shared"
    # Gain on the fixed sinusoidal readout positions; optionally learned.
    position_scale: float = 1.0
    learnable_position_scale: bool = False
    # "last" copies the last observed V-JEPA frame into every future slot; "warp" moves
    # it along a constant-velocity estimate from the observed branch's own (detached)
    # L15H7 K field plus a zero-initialized learned residual flow.
    future_init: str = "last"
    future_motion_radius: int = 3
    future_motion_temperature: float = 0.02
    # "memory": future temporal attention sees future slots only; "observed": it also
    # reads the (detached) decoded observed field at the same cell.
    future_context: str = "memory"
    # Future decoder depth; 0 keeps decoder_layers.
    future_layers: int = 0

    def validate(self):
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
        if min(self.image_size, self.grid_size, self.window_size) < 1:
            raise ValueError("Image, readout grid and window sizes must be positive")
        if self.image_size % 16 or self.grid_size % self.window_size:
            raise ValueError(
                "Image must divide by patch16; readout grid must divide by window size"
            )
        if self.dim < 12 or self.attention_heads < 1 or self.dim % self.attention_heads:
            raise ValueError("Invalid token width / attention head count")
        if min(self.language_layers, self.decoder_layers, self.head_layers) < 1:
            raise ValueError("Language, future decoder and head layers must be positive")
        if self.channels < 1 or self.text_dim < 1 or self.window_size < 2:
            raise ValueError("Invalid output/text/window dimensions")
        if self.future_query not in {"shared", "per_time"}:
            raise ValueError("future_query must be shared or per_time")
        if not 0 <= self.position_scale < float("inf"):
            raise ValueError("position_scale must be finite and nonnegative")
        if self.future_init not in {"last", "warp"}:
            raise ValueError("future_init must be last or warp")
        if self.future_init == "warp" and self.task != "shared":
            raise ValueError("future_init=warp reads the shared seg_l15h7 head")
        if self.future_motion_radius < 1 or not self.future_motion_temperature > 0:
            raise ValueError("future motion radius and temperature must be positive")
        if self.future_context not in {"memory", "observed"}:
            raise ValueError("future_context must be memory or observed")
        if self.future_layers < 0:
            raise ValueError("future_layers must be nonnegative")


@dataclasses.dataclass
class TeacherConfig:
    model_root: str = "../models/Wan2.1-VACE-1.3B"
    text_root: str = "../models/Wan2.1-T2V-1.3B-Diffusers"
    image_size: int = 512
    sigma: float = 0.8
    scheduler_steps: int = 40
    flow_shift: float = 3.0
    guidance: float = 5.0
    denoise_updates: int = 1
    denoise_reference_updates: int = 13
    future_depth_status: str = "unvalidated_protocol_transfer"
    anchors_per_time: int = 64
    beam_size: int = 4
    beam_topk: int = 32
    beam_max_step: float = 4.0
    motion_weight: float = 0.15
    local_weight: float = 0.25
    region_weight: float = 0.08
    acceleration_weight: float = 0.05
    descriptor_ema: float = 0.1
    regularize_max_step: float = 3.0
    regularize_max_acceleration: float = 1.5
    # Full-instruction tracking seeds: "max" content-token response, or
    # "spatial_normalized_max" (each token divided by its own mean over cells).
    seed_rule: str = "max"

    def validate(self):
        if self.image_size < 16 or self.image_size % 16 or not 0 < self.sigma <= 1:
            raise ValueError("Invalid teacher resolution / sigma")
        if not 1 <= self.denoise_updates <= self.denoise_reference_updates < self.scheduler_steps:
            raise ValueError("Invalid denoising schedule")
        if min(self.anchors_per_time, self.beam_size, self.beam_topk) < 1:
            raise ValueError("Invalid trajectory decoding settings")
        if self.seed_rule not in {"max", "spatial_normalized_max"}:
            raise ValueError("seed_rule must be max or spatial_normalized_max")


@dataclasses.dataclass
class TrainConfig:
    dataset_root: str = "../datasets/libero"
    manifest: str = "../runs/vace_latent_data/episodes.jsonl"
    text_cache: str = "../runs/vace_latent_data/text.pt"
    offline_manifest: str | None = None
    teacher_cache_dir: str = "../cache/vace_teacher_libero8"
    teacher_cache_limit_bytes: int = 50_000_000_000
    tos_config: str = "/root/.tosutilconfig"
    tos_endpoint: str = "tos-cn-shanghai.ivolces.com"
    output: str = "../runs/vace_structured_online_step1"
    steps: int = 10000
    batch_size: int = 32
    teacher_batch_size: int = 8
    accumulation_steps: int = 1
    windows_per_episode_camera: int = 4
    workers: int = 2
    seed: int = 42
    backbone_lr: float = 5e-5
    head_lr: float = 5e-4
    weight_decay: float = 0.05
    warmup_steps: int = 100
    grad_clip: float = 1.0
    parallelism: str = "fsdp2"
    save_every: int = 100
    log_every: int = 10
    relation_queries: int = 256
    relation_temperature: float = 0.07
    future_weight: float = 1.0
    trajectory_weight: float = 0.25
    consistency_weight: float = 0.05
    cycle_weight: float = 0.01
    depth_spatial_weight: float = 0.01
    # "uniform" relation queries, or "teacher_seeds": relation_seed_fraction of each
    # sample's queries come from teacher tracking seeds, and seeds whose teacher beam
    # path moves >= relation_moving_min_shift cells weigh relation_moving_weight.
    relation_sampling: str = "uniform"
    relation_seed_fraction: float = 0.5
    relation_moving_weight: float = 2.0
    relation_moving_min_shift: float = 1.0
    tracking_diagnostics: bool = False
    # > 0 adds a per-content-word spatial-map KL to the semantic head.
    semantic_spatial_weight: float = 0.0
    # Optional {camera: weight} for the depth loss, e.g. wrist only; None = all 1.
    depth_camera_weights: dict[str, float] | None = None
    # Weight of the dense teacher-relation KL inside the tracking loss (0 = foreground
    # beam-track distillation only) and width (grid cells) of the path-loss target.
    relation_weight: float = 1.0
    trajectory_sigma: float = 0.65
    # "relation": teacher Q-K relation KL; "descriptor": cosine to the teacher's Q and K
    # vectors (tracked K-K at readout, as in HEFT).  Path/cycle terms still apply by weight.
    tracking_target: str = "relation"
    head_weights: dict[str, float] = dataclasses.field(
        default_factory=lambda: {
            name: 2.0 if name == "tracking" else 1.0 for name in HEAD_TASKS
        }
    )

    def validate(self):
        if (
            min(
                self.steps,
                self.batch_size,
                self.teacher_batch_size,
                self.accumulation_steps,
                self.windows_per_episode_camera,
                self.save_every,
                self.log_every,
            )
            < 1
        ):
            raise ValueError("Training counts must be positive")
        if self.workers < 0 or self.parallelism not in {"fsdp2", "ddp"}:
            raise ValueError("Invalid workers / parallelism")
        if self.teacher_cache_limit_bytes <= 0:
            raise ValueError("Teacher cache limit must be positive")
        if min(self.backbone_lr, self.head_lr, self.grad_clip) <= 0 or self.warmup_steps < 0:
            raise ValueError("Learning rates and gradient clip must be positive")
        if self.relation_queries < 1 or self.relation_temperature <= 0:
            raise ValueError("Invalid relation sampling / temperature")
        if (
            min(
                self.future_weight,
                self.trajectory_weight,
                self.consistency_weight,
                self.cycle_weight,
                self.depth_spatial_weight,
            )
            < 0
        ):
            raise ValueError("Loss weights must be nonnegative")
        if set(self.head_weights) != set(HEAD_TASKS) or any(
            not isinstance(value, (int, float)) or not 0 <= value < float("inf")
            for value in self.head_weights.values()
        ):
            raise ValueError(
                "head_weights must name all five heads with finite nonnegative weights"
            )
        if not any(self.head_weights.values()):
            raise ValueError("At least one head must have a positive weight")
        if self.relation_sampling not in {"uniform", "teacher_seeds"}:
            raise ValueError("relation_sampling must be uniform or teacher_seeds")
        if not 0 <= self.relation_seed_fraction <= 1 or self.relation_moving_weight < 0:
            raise ValueError("Invalid seed fraction / moving weight")
        if self.relation_moving_min_shift < 0:
            raise ValueError("relation_moving_min_shift must be nonnegative")
        if not 0 <= self.relation_weight < float("inf"):
            raise ValueError("relation_weight must be finite and nonnegative")
        if not 0 < self.trajectory_sigma < float("inf"):
            raise ValueError("trajectory_sigma must be finite and positive")
        if self.tracking_target not in {"relation", "descriptor"}:
            raise ValueError("tracking_target must be relation or descriptor")
        if not 0 <= self.semantic_spatial_weight < float("inf"):
            raise ValueError("semantic_spatial_weight must be finite and nonnegative")
        if self.depth_camera_weights is not None and (
            not self.depth_camera_weights
            or any(
                not isinstance(value, (int, float)) or not 0 <= value < float("inf")
                for value in self.depth_camera_weights.values()
            )
        ):
            raise ValueError("depth_camera_weights must map cameras to finite nonnegative weights")

    def relation_sampling_options(self):
        if self.relation_sampling == "uniform":
            return None
        return {
            "seed_fraction": self.relation_seed_fraction,
            "moving_weight": self.relation_moving_weight,
            "min_shift": self.relation_moving_min_shift,
        }


@dataclasses.dataclass
class PipelineConfig:
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    teacher: TeacherConfig = dataclasses.field(default_factory=TeacherConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)

    def validate(self):
        self.model.validate()
        self.teacher.validate()
        self.train.validate()
        if self.model.grid_size != self.teacher.image_size // 16:
            raise ValueError("Student readout grid must match native teacher token grid")

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        raw = json.loads(path.read_text())
        if set(raw) - {"model", "teacher", "train"}:
            raise ValueError("Unknown top-level pipeline options")
        config = cls(
            **{
                name: kind(**raw.get(name, {}))
                for name, kind in (
                    ("model", ModelConfig),
                    ("teacher", TeacherConfig),
                    ("train", TrainConfig),
                )
            }
        )
        config.resolve_paths()
        config.validate()
        return config

    def resolve_paths(self):
        # Relative paths in the shipped recipe are relative to the heft checkout.
        base = Path(__file__).resolve().parents[3]
        for owner, names in (
            (self.model, ("vjepa_root", "vjepa_weights")),
            (self.teacher, ("model_root", "text_root")),
            (self.train, ("dataset_root", "manifest", "text_cache", "output",
                          "offline_manifest", "teacher_cache_dir", "tos_config")),
        ):
            for name in names:
                if getattr(owner, name) is None:
                    continue
                value = Path(getattr(owner, name)).expanduser()
                setattr(
                    owner,
                    name,
                    str(value.resolve() if value.is_absolute() else (base / value).resolve()),
                )


def timeline(device=None):
    import torch

    return {
        "observed": {
            "native_time": torch.arange(12, device=device),
            "nominal_rgb_time": torch.arange(12, device=device) * 4 - WARMUP_FRAMES,
            "valid": torch.arange(12, device=device) >= 4,
            "anchor_native": torch.tensor(OBSERVED_ANCHORS, device=device),
            "anchor_rgb": torch.tensor((0, 8, 16, 24), device=device),
        },
        "future": {
            "native_time": torch.arange(13, device=device),
            "nominal_rgb_time": torch.arange(13, device=device) * 4,
            "valid": torch.arange(13, device=device) >= 8,
            "anchor_native": torch.tensor(FUTURE_ANCHORS, device=device),
            "anchor_rgb": torch.tensor((33, 41), device=device),
        },
    }


def shared_key(task, key):
    return f"{key}_{task.removeprefix('seg_')}" if task.startswith("seg_") else key


def head_view(branch, task):
    """Expose shared outputs using the independent head's loss/readout contract."""
    if task.startswith("seg_"):
        return {"seg_k": branch[shared_key(task, "seg_k")]}
    if task == "depth":
        return {"depth": branch["depth"], "seg_k": branch["seg_k_l22h9"]}
    return branch
