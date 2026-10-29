"""StructWAM: video DiT + ActionDiT + structure expert in one joint attention.

The tri-system layout with the VLM understanding stream replaced by the VACE student's
structure stream ("geometry for the easy part, imagination for the hard part"):

* structure expert  — tokens from the frozen student (maps, keypoint tracks, V-JEPA2 trunk);
  reads video keys so the world model can refine longer-range tracks.
* phase gate        — predicts contact phase; routes the action stream's attention towards
  structure far from contact and towards the video (wrist tile) near contact.
* sketch head       — structure-only action chunk; warm-starts the action denoiser.
* video-free rows   — opt-in (``structure.action_video_dropout`` while training,
  ``inference.structure_only_action`` at inference): action and structure rows without
  video keys, so the action has to come from the structure stream.
* video stream      — supervised in latent space only, with a structure-weighted and
  difficulty-weighted loss, track consistency, and REPA-style alignment to the student trunk;
  at inference it takes 1-2 denoising steps (more near contact) and is never decoded.

Training-only inputs (prepared in ``prepare_inputs``): ``structure`` (student features per
camera) and ``phase_target``.  Deploy inputs: ``structure_frames`` (raw frame history kept by
``WAMPolicy``).
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from openwam.model.action_backbone.separate_action_dit import ActionDiT
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.registry import register_architecture
from openwam.model.architectures.tri_system.struct_driver import StructMoTDriver
from openwam.model.architectures.tri_system.struct_expert import StructureExpert, StructureExpertConfig
from openwam.model.architectures.utils.common import resolve_bridge_layers
from openwam.model.architectures.utils.mask_modes import ACTION_SEES_VIDEO
from openwam.structure import losses as structure_losses
from openwam.structure.layout import CanvasLayout
from openwam.structure.phase import PhaseGate
from openwam.structure.routing import RoutingBias
from openwam.structure.sketch import SketchHead, masked_action_mse, shape_schedule, warm_start
from openwam.structure.student import StructureStudent, StudentConfig, map_channel_names, word_slots
from openwam.structure.tokens import GROUPS

logger = logging.getLogger(__name__)

STRUCTURE_KEYS = ("structure", "phase_target")


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _as_dict(cfg) -> dict:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    try:
        from omegaconf import OmegaConf

        return dict(OmegaConf.to_container(cfg, resolve=True))
    except Exception:  # plain namespace-like objects
        return {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}


def _match_batch(features: dict, batch: int) -> dict:
    """Repeat single-sample structure features to ``batch`` (e.g. CFG-merged generation)."""
    out = {}
    for camera, value in features.items():
        out[camera] = {key: (tensor if tensor.shape[0] == batch else tensor.expand(batch, *tensor.shape[1:]))
                       for key, tensor in value.items()}
    return out


@register_architecture(
    "tri_system_struct",
    status="supported",
    note="StructWAM: video DiT + ActionDiT + VACE-student structure expert, phase-gated routing.",
    framework="tri_system",
    variant="struct_expert",
)
class TriSystemStructArchitecture(BaseWAMArchitecture):
    _NEVER_FREEZE = frozenset({"structure_expert"})

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.structure_expert = None
        self._driver = None
        self._aux: dict = {}
        self._loss_context: dict = {}
        self._student = None
        self._saliency_channels = None
        self._loss_calls = 0
        structure = _as_dict(_cfg_get(cfg, "structure", {}))
        self.layout = CanvasLayout.for_mode(str(structure.get("canvas_layout", "lshape")))
        self.structure_cfg = structure
        self.loss_cfg = _as_dict(structure.get("losses", {}))
        self.inference_cfg = _as_dict(structure.get("inference", {}))
        self.student_cfg = StudentConfig(**_as_dict(structure.get("student", {}))) if structure.get("student") else None
        if cfg is None:
            return
        vb = self.video_backbone
        cfg = dict(cfg) if isinstance(cfg, dict) else {k: v for k, v in cfg.items()}
        cfg.setdefault("bridge_interval", 1)

        expert_cfg = _as_dict(_cfg_get(cfg, "structure_expert", {}))
        cameras = tuple(structure.get("cameras", ("head", "wrist")))
        self.cameras = cameras
        if self.student_cfg is None:
            raise ValueError("architecture.structure.student is not configured")
        self.structure_expert = StructureExpert(
            StructureExpertConfig(dim=int(expert_cfg.get("dim", 512)), ffn_dim=int(expert_cfg.get("ffn_dim", 2048)),
                                  num_layers=vb.num_layers, eps=float(expert_cfg.get("eps", 1e-5)), cameras=cameras,
                                  map_channels=len(map_channel_names(self.student_cfg)),
                                  word_slots=word_slots(self.student_cfg),
                                  groups=tuple(expert_cfg.get("groups", GROUPS))),
            wan_dim=vb.dim,
            wan_num_heads=vb.num_heads,
        )
        dim = int(expert_cfg.get("dim", 512))

        action_hidden = int(_cfg_get(cfg, "dim", 1024))
        text_dim = int(_cfg_get(cfg, "text_dim", getattr(vb, "text_dim", None) or 4096))
        bridge = resolve_bridge_layers(cfg, num_layers=vb.num_layers)
        self._init_proprio_context(cfg, text_dim=text_dim)
        action_dim = int(_cfg_get(cfg, "action_dim", 20))
        self.action_backbone = ActionDiT(
            action_dim=action_dim,
            dim=action_hidden,
            ffn_dim=int(_cfg_get(cfg, "ffn_dim", action_hidden * int(_cfg_get(cfg, "ffn_dim_multiplier", 4)))),
            num_heads=int(_cfg_get(cfg, "num_heads", vb.num_heads)),
            num_layers=len(bridge),
            video_dim=vb.dim,
            bridge_layers=bridge,
            variant="joint_self_attn",
            attn_head_dim=int(_cfg_get(cfg, "attn_head_dim", vb.head_dim)),
            text_dim=text_dim,
            shift_action=_cfg_get(cfg, "shift_action"),
        )

        state_dim = int(_cfg_get(cfg, "state_dim", 0)) if _cfg_get(cfg, "use_proprioception", False) else None
        self.phase_gate = PhaseGate(dim, state_dim)
        self.sketch_head = SketchHead(dim, int(structure.get("horizon", 32)), action_dim, cameras, state_dim)
        routing = _as_dict(structure.get("routing", {}))
        self.routing = RoutingBias(float(routing.get("init_structure", 1.0)), float(routing.get("init_video", 1.0)),
                                   float(routing.get("init_wrist", 0.5))) if routing.get("enabled", True) else None
        align_layer = structure.get("align_layer")
        self.align_layer = vb.num_layers // 2 if align_layer is None else int(align_layer)
        self.align_projector = nn.Sequential(nn.LayerNorm(vb.dim), nn.Linear(vb.dim, 2048), nn.SiLU(),
                                             nn.Linear(2048, 1024))
        self._driver_kwargs = {
            "mot_checkpoint_mixed_attn": bool(_cfg_get(cfg, "mot_checkpoint_mixed_attn", True)),
            "attention_mask_mode": str(_cfg_get(cfg, "attention_mask_mode", ACTION_SEES_VIDEO)),
            "video_attention_mask_mode": str(_cfg_get(cfg, "video_attention_mask_mode", "first_frame_causal")),
            "structure_reads_video": bool(structure.get("structure_reads_video", True)),
            "video_detach": tuple(structure.get("video_detach", ())),
            "align_layer": None if float(self.loss_cfg.get("lambda_align", 0.0)) == 0 else int(self.align_layer),
        }
        self.build_driver()

    # ----- components -----------------------------------------------------------

    def build_driver(self) -> StructMoTDriver:
        self._driver = StructMoTDriver(self.video_backbone, self.action_backbone, self.structure_expert,
                                       routing=self.routing, **self._driver_kwargs)
        return self._driver

    @property
    def student(self) -> StructureStudent:
        if self._student is None:
            if self.student_cfg is None:
                raise ValueError("architecture.structure.student is not configured")
            self._student = StructureStudent(self.student_cfg, self.device)
        return self._student

    @property
    def structure_history_length(self) -> int:
        return int(self.student_cfg.history) if self.student_cfg is not None else 0

    def saliency_channels(self):
        if self._saliency_channels is None:
            names = self.student.map_channels
            self._saliency_channels = [names.index(n) for n in structure_losses.SALIENCY_CHANNELS if n in names]
        return self._saliency_channels

    def freeze_modules(self, names: list[str]) -> list[str]:
        rejected = self._NEVER_FREEZE & set(names)
        if rejected:
            raise ValueError(f"tri_system_struct: refusing to freeze {rejected}")
        return super().freeze_modules(names)

    def set_dtype_device(self, dtype, device):
        super().set_dtype_device(dtype, device)
        for module in (self.structure_expert, self.phase_gate, self.sketch_head, self.routing, self.align_projector):
            if module is not None:
                module.to(dtype=dtype, device=device)

    def _wrist_video_tokens(self, vstate) -> Optional[Tensor]:
        if "wrist" not in self.cameras:
            return None
        frames, height, width = int(vstate.grid_frames), int(vstate.grid_height), int(vstate.grid_width)
        self.layout.check_grid(height, width)
        region = self.layout.region_mask("wrist", height, width, device=self.device).flatten()
        return region.repeat(frames)

    # ----- training inputs ------------------------------------------------------

    @torch.no_grad()
    def prepare_inputs(self, batch: list[dict]) -> dict:
        if isinstance(batch, dict):
            batch = [batch]
        inputs = super().prepare_inputs(batch)
        histories = [sample.get("structure_frames") or {} for sample in batch]
        frames = {}
        for camera in self.cameras:
            clips = [history.get(camera) for history in histories]
            if all(clip is not None for clip in clips):
                frames[camera] = torch.from_numpy(np.stack(clips))
        inputs["structure"] = self.student.encode(frames, [sample["prompt"] for sample in batch])
        if all("phase_chunk" in sample for sample in batch):
            inputs["phase_target"] = torch.tensor([float(s["phase_chunk"]) for s in batch], device=self.device)
        return inputs

    # ----- forward --------------------------------------------------------------

    def _video_drop(self, batch: int, device) -> Optional[Tensor]:
        """Samples whose action and structure rows see no video keys: each with probability
        ``structure.action_video_dropout`` while training, all of them at inference when
        ``inference.structure_only_action``; ``None`` otherwise (no random draw, masks unchanged)."""
        if self.training:
            p = float(self.structure_cfg.get("action_video_dropout", 0.0))
        else:
            p = 1.0 if self.inference_cfg.get("structure_only_action", False) else 0.0
        if p <= 0:
            return None
        if p >= 1:
            return torch.ones(batch, dtype=torch.bool, device=device)
        return torch.rand(batch, device=device) < p

    def _gate(self, pooled: Tensor, proprio: Optional[Tensor], phase_target: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        logit = self.phase_gate(pooled, proprio)
        gate = torch.sigmoid(logit.float())
        forcing = float(self.structure_cfg.get("gate_teacher_forcing", 0.0))
        if self.training and phase_target is not None and forcing > 0:
            use_target = torch.rand_like(gate) < forcing
            gate = torch.where(use_target, phase_target.to(gate.dtype), gate)
        return logit, gate

    def forward(
        self,
        noisy_actions: Optional[Tensor],
        action_timestep: Optional[Tensor],
        *,
        proprio: Optional[Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **pipeline_inputs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        vb, ab, sb = self.video_backbone, self.action_backbone, self.structure_expert
        structure = pipeline_inputs.pop("structure", None)
        phase_target = pipeline_inputs.pop("phase_target", None)
        raw_proprio = proprio
        pipeline_inputs = self._append_proprio_context_token(dict(pipeline_inputs), proprio)
        action_context = pipeline_inputs.get("context")
        action_context_mask = pipeline_inputs.get("context_mask")
        if action_context is not None and action_context_mask is None and pipeline_inputs.get("seq_lens") is not None:
            seq_lens = pipeline_inputs["seq_lens"].to(device=action_context.device)
            positions = torch.arange(action_context.shape[1], device=action_context.device)
            action_context_mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
        pipeline_inputs.setdefault("force_per_token_t_mod", True)
        pipeline_inputs.setdefault("zero_clean_prefix_t_mod", True)
        vstate = vb.prepare(
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            **pipeline_inputs,
        )
        if noisy_actions is None or ab is None:
            for block_id in range(vb.num_layers):
                vstate = vb.run_block(block_id, vstate)
            return vb.finalize(vstate), None
        if structure is None:
            raise ValueError("tri_system_struct forward with actions requires `structure` features")

        batch = noisy_actions.shape[0]
        structure = _match_batch(structure, batch)
        astate = ab.prepare_state(
            noisy_actions,
            action_timestep,
            context=action_context,
            context_mask=action_context_mask,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        sstate = sb.prepare_state(structure, batch=batch, dtype=vstate.hidden_states.dtype)
        logit, gate = self._gate(sstate.pooled, raw_proprio, phase_target)
        driver = self._driver or self.build_driver()
        driver.set_forward_context(gate if self.routing is not None else None, self._wrist_video_tokens(vstate),
                                   video_drop=self._video_drop(batch, gate.device))
        vstate, astate, sstate = driver.run_joint_loop(
            vstate,
            astate,
            sstate,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        self._aux = {
            "gate_logit": logit,
            "phase_target": phase_target,
            "sketch": self.sketch_head(sstate.pooled, structure, raw_proprio),
            "hidden": driver.captured_hidden,
            "grid": (int(vstate.grid_frames), int(vstate.grid_height), int(vstate.grid_width)),
            "structure": structure,
        }
        return vb.finalize(vstate), ab.extract_prediction(astate)

    # ----- losses -----------------------------------------------------------------

    def compute_loss(self, *, actions: Optional[Tensor] = None, lambda_video: float = 1.0,
                     lambda_action: float = 1.0, **inputs) -> dict:
        target_actions = actions if actions is not None else inputs.get("actions")
        self._loss_context = {"actions": target_actions, "action_is_pad": inputs.get("action_is_pad")}
        self._aux = {}
        result = super().compute_loss(actions=actions, lambda_video=lambda_video, lambda_action=lambda_action,
                                      **inputs)
        aux, terms = self._aux, {}
        if aux.get("phase_target") is not None:
            terms["phase"] = F.binary_cross_entropy_with_logits(aux["gate_logit"].float(),
                                                                aux["phase_target"].float())
        if aux.get("sketch") is not None and target_actions is not None:
            terms["sketch"] = self._sketch_error(aux["sketch"]).mean()
        if aux.get("hidden") is not None:
            hidden = structure_losses.hidden_grid(aux["hidden"], *aux["grid"])
            terms["align"] = structure_losses.alignment_loss(hidden, self.align_projector, aux["structure"],
                                                             self.layout)
        if aux.get("track") is not None:
            terms["track"] = aux["track"]
        loss = result["loss"]
        for name, value in terms.items():
            weight = float(self.loss_cfg.get(f"lambda_{name}", 0.0))
            if weight > 0:
                loss = loss + weight * value
            result[f"loss_{name}"] = value.detach()
        result["loss"] = loss
        self._log_terms(result, aux)
        return result

    def _log_terms(self, result: dict, aux: dict) -> None:
        """The OpenWAM trainer logs total / video / action only; the structure terms go to the log here."""
        self._loss_calls += 1
        every = int(self.structure_cfg.get("log_every", 50))
        if every <= 0 or self._loss_calls % every:
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank():
            return
        parts = [f"{key[5:]}={float(value):.4f}" for key, value in result.items()
                 if key.startswith("loss_") and key not in ("loss_video", "loss_action")]
        if aux.get("gate_logit") is not None:
            parts.append(f"gate_mean={float(torch.sigmoid(aux['gate_logit'].float()).mean()):.3f}")
        if self.routing is not None:
            weights = self.routing.weights.detach().float().tolist()
            parts.append("routing=" + "/".join(f"{w:.3f}" for w in weights))
        logger.info("structwam call %d: %s", self._loss_calls, " ".join(parts))

    def _sketch_error(self, sketch: Tensor) -> Tensor:
        target = self._loss_context["actions"].to(sketch.device)
        if target.dim() == 2:
            target = target.unsqueeze(0)
        return masked_action_mse(sketch, target, self._loss_context.get("action_is_pad"))

    def _video_skip(self, noise_pred: Tensor, inputs: dict) -> int:
        """Leading conditioning latents excluded from the video loss (mirrors the base rule)."""
        num_clean_prefix = int(inputs.get("num_clean_prefix_frames", 0) or 0)
        video_is_pad = inputs.get("video_is_pad")
        if inputs.get("first_frame_latents") is not None:
            return max(num_clean_prefix, 1)
        if num_clean_prefix > 0:
            return num_clean_prefix + 1
        if video_is_pad is not None and video_is_pad.shape[-1] < noise_pred.shape[2]:
            return noise_pred.shape[2] - video_is_pad.shape[-1]
        return 0

    def _compute_video_loss(self, noise_pred, target, timestep_ids, inputs, device):
        structure = inputs.get("structure")
        if structure is None:
            return super()._compute_video_loss(noise_pred, target, timestep_ids, inputs, device)
        vb = self.video_backbone
        structure = _match_batch(structure, noise_pred.shape[0])
        pixel = structure_losses.video_weights(structure, self.saliency_channels(), self.layout,
                                               noise_pred.shape[-2:], float(self.loss_cfg.get("video_saliency", 0.0)))
        sketch = self._aux.get("sketch")
        sketch_error = self._sketch_error(sketch).detach() if sketch is not None and \
            self._loss_context.get("actions") is not None else None
        phase = inputs.get("phase_target")
        if phase is None:
            phase = torch.zeros(noise_pred.shape[0], device=noise_pred.device)
        sample = structure_losses.difficulty(phase, structure_losses.mean_track_similarity(structure), sketch_error,
                                             self.loss_cfg.get("difficulty", (0.0, 0.0, 0.0)))
        timestep_weight = vb.scheduler.linear_timesteps_weights[timestep_ids].to(dtype=torch.float32, device=device)
        loss = structure_losses.weighted_video_mse(noise_pred, target, timestep_weight,
                                                   n_skip=self._video_skip(noise_pred, inputs),
                                                   video_is_pad=inputs.get("video_is_pad"), pixel_weight=pixel,
                                                   sample_weight=sample)
        if float(self.loss_cfg.get("lambda_track", 0.0)) > 0 and noise_pred.shape[2] > 1:
            sigma = vb.scheduler.sigmas[timestep_ids].to(device=noise_pred.device, dtype=torch.float32)
            x0_pred = inputs["latents"].float() - sigma.view(-1, 1, 1, 1, 1) * noise_pred.float()
            self._aux["track"] = structure_losses.track_consistency(x0_pred, inputs["input_latents"], structure,
                                                                    self.layout)
        return loss

    # ----- inference --------------------------------------------------------------

    @torch.no_grad()
    def generate(self, schedule, prompt: str, *, structure_frames: Optional[dict] = None,
                 proprio: Optional[Tensor] = None, num_frames: int = 49, action_num_frames: Optional[int] = None,
                 seed: int = 42, **kwargs) -> dict:
        if structure_frames is None:
            raise ValueError("tri_system_struct.generate needs `structure_frames` (WAMPolicy keeps the history)")
        self.eval()
        frames = {camera: torch.from_numpy(np.asarray(clip))[None] for camera, clip in structure_frames.items()
                  if camera in self.cameras}
        features = self.student.encode(frames, [prompt])
        state_proprio = torch.as_tensor(proprio).reshape(1, -1).to(self.device, self.dtype) if proprio is not None \
            else None
        pooled = self.structure_expert.tokenizer(features, 1, dtype=self.dtype)["pooled"]
        gate = float(torch.sigmoid(self.phase_gate(pooled, state_proprio).float()))
        near = gate >= float(self.inference_cfg.get("near_threshold", 0.5))
        video_steps = int(self.inference_cfg.get("video_steps_near" if near else "video_steps_far", 2 if near else 1))
        action_frames = int(action_num_frames if action_num_frames is not None else num_frames)
        start_t = self.inference_cfg.get("sketch_start_t")
        sketch = self.sketch_head(pooled, features, state_proprio)
        if sketch.shape[1] != action_frames - 1:
            start_t = None
        shaped = shape_schedule(schedule, video_steps=video_steps, action_start_t=start_t)
        initial = None
        if start_t is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            initial = warm_start(sketch.to(self.dtype), self.action_scheduler, shaped[0][1], generator)
        kwargs.pop("decode_video", None)
        result = super().generate(shaped, prompt, num_frames=num_frames, action_num_frames=action_num_frames,
                                  seed=seed, proprio=proprio, initial_action_latents=initial, decode_video=False,
                                  structure=features, **kwargs)
        result["structure_gate"] = gate
        result["video_steps"] = video_steps
        return result


__all__ = ["TriSystemStructArchitecture"]
