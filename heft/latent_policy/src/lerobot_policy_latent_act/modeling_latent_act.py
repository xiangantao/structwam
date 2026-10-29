"""Thin ACT subclasses: frozen visual tokens, task identity, and causal observation history."""

import torch
from lerobot.policies.act.modeling_act import ACT, ACTPolicy, create_sinusoidal_pos_embedding
from lerobot.policies.pi05.memory import temporal_sinusoidal_embedding
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from .configuration_latent_act import LatentACTConfig
from .data import canonical_task
from .encoders import make_encoder
from .history import CameraHistory


class LatentACT(ACT):
    def __setattr__(self, name, value):
        # Frozen runtimes may contain nn.Modules; never register them in ACT's module tree.
        if name == "_visual_runtime":
            object.__setattr__(self, name, value)
            return
        super().__setattr__(name, value)

    def __init__(self, config):
        if not hasattr(ACT, "_visual_tokens"):
            raise RuntimeError(
                "Apply latent_policy/patches/lerobot-act.patch to the local LeRobot tree"
            )
        super().__init__(config)
        if config.checkpoint_encoder:
            self.encoder.layers = nn.ModuleList(
                checkpoint_wrapper(layer) for layer in self.encoder.layers
            )
        # This is an external frozen runtime, deliberately excluded from parameters/state_dict.
        object.__setattr__(self, "_visual_runtime", None)
        object.__setattr__(self, "_runtime_device", None)

    def _build_visual_backbone(self, config):
        return config.visual_dim

    def _visual_tokens(self, images):
        device = self.encoder_img_feat_input_proj.weight.device
        if self._visual_runtime is None:
            # Frozen-model construction must not consume the trainable ACT's RNG stream.
            with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                runtime = make_encoder(self.config, device)
            # A runtime may wrap nn.Modules, but it must remain outside ACT's module tree.
            object.__setattr__(self, "_visual_runtime", runtime)
            object.__setattr__(self, "_runtime_device", device)
        elif device != self._runtime_device:
            raise RuntimeError(
                "Close the frozen encoder runtime before moving a used policy to another device"
            )
        # Enforce frozen extraction even when the training loop uses autocast.
        with torch.no_grad(), torch.autocast(device.type, enabled=False):
            pending = getattr(self, "_pending_episode_indices", None)
            if pending is None:
                maps = self._visual_runtime.encode(images)
            else:
                maps = self._visual_runtime.encode(images, episode_indices=pending)
                object.__setattr__(self, "_pending_episode_indices", None)
        if len(images) != len(self.config.camera_keys):
            raise ValueError("Expected one image history for each configured camera")
        if len(maps) != len(self.config.camera_keys):
            raise ValueError("Frozen encoder returned the wrong number of cameras")
        expected = getattr(self.config, "visual_shape", None)
        if expected is None:
            expected = {
                "wan": (16, 896, 14, 14),
                "vjepa2": (8, 1024, 16, 16),
                "wan_jepa": (8, 1152, 16, 16),
                "distilled": (8, 1152, 16, 16),
            }.get(getattr(self.config, "encoder", None))
        if expected is None or len(expected) != 4:
            raise ValueError("Frozen encoder shape contract is missing or invalid")
        expected = tuple(expected)
        tokens, positions = [], []
        camera_pos = create_sinusoidal_pos_embedding(len(maps), self.config.dim_model).to(device)
        for camera, features in enumerate(maps):
            if features.ndim != 5:
                raise ValueError(f"Unexpected frozen feature rank: {tuple(features.shape)}")
            batch, frames, channels, height, width = features.shape
            image_shape = getattr(images[camera], "shape", ())
            if not image_shape or batch != image_shape[0]:
                raise ValueError(f"Frozen feature batch does not match camera {camera}")
            if tuple(features.shape[1:]) != expected:
                raise ValueError(f"Unexpected frozen feature shape: {tuple(features.shape)}")
            # The ACT 1x1 projection is shared over cameras, space, and time.
            # Clone with inference mode disabled: some frozen adapters return inference tensors,
            # which cannot be consumed by a trainable Conv2d backward pass.
            with torch.inference_mode(False):
                features = features.detach().clone().to(
                    dtype=self.encoder_img_feat_input_proj.weight.dtype
                )
            projected = self.encoder_img_feat_input_proj(
                features.reshape(-1, channels, height, width)
            )
            projected = projected.reshape(batch, frames, self.config.dim_model, height, width)
            spatial = self.encoder_cam_feat_pos_embed(features[:, 0]).to(projected.dtype)
            temporal = temporal_sinusoidal_embedding(
                self.config.history_frames,
                self.config.dim_model,
                device=device,
                dtype=projected.dtype,
            )
            if self.config.history_frames % frames:
                raise ValueError(
                    "Frozen feature frame count must divide the configured history length"
                )
            # V-JEPA2 and distilled tokens correspond to adjacent RGB-frame pairs; average
            # each pair's input-time codes while leaving camera identity separate below.
            temporal = temporal.reshape(frames, self.config.history_frames // frames, -1).mean(1)
            position = spatial[:, None] + temporal[None, :, :, None, None]
            position = position + camera_pos[camera][None, None, :, None, None]
            tokens.append(
                projected.permute(1, 3, 4, 0, 2).reshape(-1, batch, self.config.dim_model)
            )
            positions.append(position.permute(1, 3, 4, 0, 2).reshape(-1, 1, self.config.dim_model))
        return torch.cat(tokens), torch.cat(positions)

    def close(self):
        if self._visual_runtime is not None:
            self._visual_runtime.close()
            object.__setattr__(self, "_visual_runtime", None)
            object.__setattr__(self, "_runtime_device", None)


class LatentACTPolicy(ACTPolicy):
    config_class = LatentACTConfig
    name = "latent_act"

    def _make_model(self, config):
        return LatentACT(config)

    def reset(self):
        super().reset()
        self._history = CameraHistory(self.config.camera_keys, self.config.history_frames)

    def _prepare_batch(self, batch):
        result = dict(batch)
        tasks = result.get("task")
        if isinstance(tasks, str):
            tasks = [tasks]
        if tasks is None or len(tasks) != result[OBS_STATE].shape[0]:
            raise ValueError(
                "Every batch row requires its task description for suite-local task conditioning"
            )
        mapping = {canonical_task(name): i for i, name in enumerate(self.config.task_names)}
        try:
            ids = [mapping[canonical_task(name)] for name in tasks]
        except KeyError as exc:
            raise ValueError(f"Task outside configured suite {self.config.suite}: {exc}") from exc
        if result[OBS_STATE].ndim != 2:
            raise ValueError("ACT requires only the current robot state [B,8]")
        result[OBS_ENV_STATE] = torch.nn.functional.one_hot(
            torch.tensor(ids, device=result[OBS_STATE].device), len(mapping)
        ).to(result[OBS_STATE].dtype)
        result[OBS_IMAGES] = [result[key] for key in self.config.camera_keys]
        if getattr(self.config, "feature_seed_mode", "fixed") == "per_clip":
            if "episode_index" not in result:
                raise ValueError(
                    "per_clip feature seeds require episode_index in the batch; it is absent, "
                    "so the run would silently fall back to one shared artifact"
                )
            # Stash on the MODEL, not on this policy: _visual_tokens runs on self.model, so an
            # attribute set here would be invisible to it (two different objects).
            object.__setattr__(
                self.model, "_pending_episode_indices",
                [int(value) for value in result["episode_index"]],
            )
        return result

    def forward(self, batch):
        # ACTPolicy creates its camera list from input_features. Keep that order canonical as well.
        return super().forward(self._ordered_batch(batch))

    def _ordered_batch(self, batch):
        result = self._prepare_batch(batch)
        for key in self.config.camera_keys:
            if result[key].ndim != 5 or result[key].shape[1] != self.config.history_frames:
                raise ValueError(
                    "Training/predict_action_chunk requires a complete causal 16-frame history"
                )
        return result

    @torch.no_grad()
    def predict_action_chunk(self, batch):
        self.eval()
        return self.model(self._ordered_batch(batch))[0]

    @torch.no_grad()
    def select_action(self, batch):
        self.eval()
        # Closed-loop rollouts build observations from the simulator, which has no dataset
        # episode index, so "one readout per episode" has no meaning there: the policy is a
        # fixed function of the current history and every step already uses a fresh noise field.
        # Use one documented constant readout instead of crashing or silently reusing the
        # training seeds. Dataset-driven callers still reach this through predict_action_chunk
        # and keep their real per-episode seeds.
        if (
            getattr(self.config, "feature_seed_mode", "fixed") == "per_clip"
            and "episode_index" not in batch
        ):
            fallback = int(getattr(self.config, "feature_seed_eval_episode", 0))
            batch = dict(batch)
            batch["episode_index"] = torch.full(
                (batch[OBS_STATE].shape[0],), fallback, dtype=torch.long
            )
        # Update on every environment step, including steps consuming an existing action chunk.
        self._history.update(batch)
        needs_chunk = self.config.temporal_ensemble_coeff is not None or not self._action_queue
        if needs_chunk:
            batch = self._history.stack(batch)
        return super().select_action(batch)

    def get_optim_params(self):
        return [{"params": [p for p in self.parameters() if p.requires_grad]}]

    def close(self):
        self.model.close()
