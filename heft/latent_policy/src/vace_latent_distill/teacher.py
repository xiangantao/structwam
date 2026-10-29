"""Frozen online VACE teacher with explicitly privileged future supervision."""

import importlib
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .config import FUTURE_ANCHORS, OBSERVED_ANCHORS, TeacherConfig
from .schedules import sparse_schedule
from .targets import validate_target


def local_scripts():
    root = Path(__file__).resolve().parents[3]
    scripts = root / "latent_policy/scripts"
    sys.path.insert(0, str(root / "diffusers/src"))
    sys.path.insert(0, str(scripts))
    return importlib.import_module("evaluate_vace_pure_noise_future")


def _teacher_opt():
    return os.environ.get("VACE_TEACHER_OPT", "1") == "1"


def _cfg_parallel():
    """Batch conditional/unconditional CFG; set VACE_CFG_PARALLEL=0 to disable."""
    return os.environ.get("VACE_CFG_PARALLEL", "1") not in ("", "0", "false", "False")


class Capture:
    def __init__(self, model, task, temporal, grid, text_mask):
        self.model, self.task, self.temporal, self.grid = model, task, temporal, grid
        self.text_mask = text_mask.bool()
        if self.text_mask.ndim == 1:
            self.text_mask = self.text_mask[None]
        self.values, self.handles, self.attentions = {}, [], []

    def __enter__(self):
        layers = {
            "seg_l22h9": {22: 9},
            "seg_l15h7": {15: 7},
            "tracking": {22: 9, 15: 7},
            "semantic": {},
            "depth": {22: 9, 16: 9},
            "shared": {22: 9, 15: 7, 16: 9},
        }[self.task]
        for layer, head in layers.items():

            def hook(_module, _inputs, output, layer=layer, head=head):
                batch = output.shape[0]
                shape = (batch, self.temporal, self.grid**2,
                         self.model.config.num_attention_heads,
                         self.model.config.attention_head_dim)
                self.values[f"k{layer}"] = (
                    output.reshape(shape)[:, :, :, head].detach().float().cpu()
                )

            self.handles.append(self.model.blocks[layer].attn1.norm_k.register_forward_hook(hook))
        if self.task in {"tracking", "shared"}:
            attention = self.model.blocks[15].attn1

            def qk(query, key, **_):
                shape = (query.shape[0], self.temporal, self.grid**2, query.shape[-1])
                self.values["track_q"] = (
                    F.normalize(query[:, 4].reshape(shape).float(), dim=-1).detach().cpu()
                )
                self.values["track_k"] = (
                    F.normalize(key[:, 4].reshape(shape).float(), dim=-1).detach().cpu()
                )

            attention._heft_capture = qk
            self.attentions.append(attention)
        if self.task in {"tracking", "semantic", "shared"}:
            attention = self.model.blocks[22].attn2

            def semantic(query, key, **_):
                pieces = []
                for begin in range(0, query.shape[2], 1024):
                    logits = (
                        torch.bmm(
                            query[:, 4, begin : begin + 1024].float(),
                            key[:, 4].float().transpose(1, 2),
                        )
                        / math.sqrt(query.shape[-1])
                    )
                    logits = logits.masked_fill(~self.text_mask[:, None], -torch.inf)
                    pieces.append(logits.softmax(-1).detach().cpu())
                self.values["text_attention"] = torch.cat(pieces, dim=1).reshape(
                    query.shape[0], self.temporal, self.grid**2, -1
                )

            attention._heft_capture = semantic
            self.attentions.append(attention)
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        for attention in self.attentions:
            del attention._heft_capture

    def result(self, index=None):
        values = self.values if index is None else {
            key: value[index] for key, value in self.values.items()
        }
        if self.task.startswith("seg_"):
            layer = 22 if self.task == "seg_l22h9" else 15
            return {"seg_k": F.normalize(values[f"k{layer}"], dim=-1)}
        if self.task == "depth":
            return {
                "depth": -values["k16"][..., 52],
                "seg_k": F.normalize(values["k22"], dim=-1),
            }
        if self.task == "semantic":
            return {"text_attention": values["text_attention"]}
        if self.task == "shared":
            return {
                "seg_k_l22h9": F.normalize(values["k22"], dim=-1),
                "seg_k_l15h7": F.normalize(values["k15"], dim=-1),
                "track_q": values["track_q"],
                "track_k": values["track_k"],
                "text_attention": values["text_attention"],
                "depth": -values["k16"][..., 52],
            }
        return values


def seed_heat(attention, content_mask, rule="max"):
    """Full-instruction seed heat ``[T, cells]`` from word attention ``[T, cells, L]``.

    ``max`` keeps the original readout: the strongest content-token response.
    ``spatial_normalized_max`` first divides each content token's map by its own
    mean over cells, so a token that is bright everywhere (e.g. a verb) cannot
    outrank a token that is concentrated on a few cells.  Only teacher attention
    and the tokenizer/stop-word content mask are used.
    """
    content = np.asarray(content_mask, dtype=bool)
    if not content.any():
        raise ValueError("Instruction has no content-token seed candidates")
    values = attention[..., content]
    if rule == "spatial_normalized_max":
        values = values / np.maximum(values.mean(-2, keepdims=True), 1e-8)
    elif rule != "max":
        raise ValueError(f"Unknown seed rule: {rule}")
    return values.max(-1)


def multi_anchor_paths(values, anchors, config, content_mask):
    demo = importlib.import_module("demo_main_prompt_k_tracking")
    temporal, cells = values["track_q"].shape[:2]
    grid = round(cells**0.5)
    count = min(config.anchors_per_time, cells)
    # This generic full-instruction seed readout is explicit: select positions
    # strongly associated with any content token, not the constant sum of all words.
    attention = values["text_attention"].numpy()
    heat = seed_heat(attention, content_mask, config.seed_rule).reshape(temporal, grid, grid)
    k_values = {22: values["k22"].numpy(), 15: values["k15"].numpy()}
    seeds, paths, masks = [], [], []
    for anchor in anchors:
        score, selected = demo.region_from_prompt_and_k(heat, k_values, anchor, grid)
        chosen = selected[np.argsort(score.reshape(-1)[selected])[-count:]]
        # Keep point count explicit; it is a per-anchor cap, not a count per object.
        chosen = np.pad(chosen, (0, count - len(chosen)), mode="edge")
        region = demo.k_region_over_time(k_values, chosen, anchor, grid)
        end = min(anchor + 3, temporal)
        q = values["track_q"][:end].numpy().reshape(-1, 128)
        k = values["track_k"][:end].numpy().reshape(-1, 128)
        path = demo.track_dense(
            q,
            k,
            chosen,
            anchor,
            grid,
            region[:end],
            topk=config.beam_topk,
            beam_size=config.beam_size,
            max_step=config.beam_max_step,
            motion_weight=config.motion_weight,
            local_weight=config.local_weight,
            region_weight=config.region_weight,
            acceleration_weight=config.acceleration_weight,
            descriptor_ema=config.descriptor_ema,
            direction="q_to_k",
        )
        path = demo.regularize_paths(
            path,
            max_step=config.regularize_max_step,
            max_acceleration=config.regularize_max_acceleration,
            passes=2,
        )
        field = np.zeros((count, temporal, 2), np.float32)
        valid = np.zeros((count, temporal), bool)
        field[:, anchor:end] = path
        valid[:, anchor:end] = True
        seeds.append(chosen)
        paths.append(field)
        masks.append(valid)
    return {
        "track_q": values["track_q"],
        "track_k": values["track_k"],
        "seeds": torch.from_numpy(np.stack(seeds)).long(),
        "paths": torch.from_numpy(np.stack(paths)),
        "path_valid": torch.from_numpy(np.stack(masks)),
    }


class VaceTeacher:
    def __init__(self, config: TeacherConfig, task, device, negative_embedding):
        source = local_scripts()
        self.config, self.task, self.device = config, task, torch.device(device)
        self.model = source.load_transformer(Path(config.model_root), self.device)
        self.vae = source.load_vae(Path(config.model_root) / "Wan2.1_VAE.pth", self.device)
        from . import npu_vae_patch

        npu_vae_patch.install(self.vae)  # no-op for output values; VACE_VAE_OPT=0 turns it off
        from diffusers import UniPCMultistepScheduler

        self.scheduler = UniPCMultistepScheduler.from_pretrained(
            Path(config.text_root) / "scheduler",
            local_files_only=True,
            flow_shift=config.flow_shift,
        )
        self.negative = negative_embedding[None].to(self.device, dtype=torch.bfloat16)
        self.scales = torch.ones(
            len(self.model.vace_blocks), device=self.device, dtype=torch.bfloat16
        )
        self.protocol = {
            "teacher": str(Path(config.model_root).resolve()),
            "task": task,
            "observed": "33 RGB +15 copies of first frame; main pure noise t1000",
            "future": "49-frame GT latent + noise; control past33 only; sparse UniPC to reference endpoint, then conditional capture",
            "requested_sigma": config.sigma,
            "denoise_updates": config.denoise_updates,
            "denoise_reference_updates": config.denoise_reference_updates,
            "future_schedule": "subsampled_unipc_reference_interval_v1",
            "future_depth_status": config.future_depth_status,
            "semantic_padding": "Padding excluded from distilled word distribution",
            "tracking_seeds": (
                "Maximum content-token attention, full instruction; no GT masks"
                if config.seed_rule == "max"
                else "Maximum of per-token spatially normalized content-token attention, "
                "full instruction; no GT masks"
            ),
            "teacher_batching": "Independent windows share batched VAE/control preparation and transformer forwards",
            "cfg_parallel": _cfg_parallel(),
        }

    def _pixels(self, rgb):
        return F.interpolate(
            rgb.to(self.device).float() / 127.5 - 1,
            size=(self.config.image_size,) * 2,
            mode="bilinear",
            align_corners=False,
        ).permute(1, 0, 2, 3)

    def _pixels_batch(self, rgb):
        """Resize a batch of uint8 videos to ``[B,C,T,H,W]`` pixels."""
        if rgb.ndim != 5 or rgb.shape[2] != 3 or rgb.dtype != torch.uint8:
            raise ValueError("Expected uint8 [B,T,3,H,W] teacher videos")
        batch, frames = rgb.shape[:2]
        resized = F.interpolate(
            rgb.to(self.device).float().flatten(0, 1) / 127.5 - 1,
            size=(self.config.image_size,) * 2,
            mode="bilinear",
            align_corners=False,
        )
        return resized.reshape(batch, frames, 3, self.config.image_size,
                               self.config.image_size).permute(0, 2, 1, 3, 4).contiguous()

    def _inputs(self, pixels, observed):
        control_pixels = torch.zeros_like(pixels)
        control_pixels[:, :observed] = pixels[:, :observed]
        inactive = self.vae.encode([control_pixels])[0]
        reactive = self.vae.encode([torch.zeros_like(control_pixels)])[0]
        mask = torch.ones((pixels.shape[1], *pixels.shape[-2:]), device=self.device)
        mask[:observed] = 0
        size = self.config.image_size
        mask = mask.view(len(mask), size // 8, 8, size // 8, 8).permute(2, 4, 0, 1, 3).flatten(0, 1)
        mask = F.interpolate(mask[None], size=inactive.shape[1:], mode="nearest-exact")[0]
        return torch.cat((inactive, reactive, mask), 0)[None].bfloat16(), inactive.shape

    def _readout_pool(self):
        if "_pool" not in self.__dict__:
            from concurrent.futures import ThreadPoolExecutor

            self.__dict__["_pool"] = ThreadPoolExecutor(max_workers=1)
        return self._pool

    def _zero_latent(self, like):
        """VAE latent of an all-zero video; deterministic, so cached per shape."""
        key = tuple(like.shape)
        cache = self.__dict__.setdefault("_zero_cache", {})
        if not _teacher_opt():
            return self.vae.encode([torch.zeros_like(like)])[0]
        if key not in cache:
            cache[key] = self.vae.encode([torch.zeros_like(like)])[0]
        return cache[key]

    def _inputs_batch(self, pixels, observed):
        """Build VACE controls for ``pixels=[B,C,T,H,W]``.

        The native VAE wrapper is list based, so each video is encoded independently;
        the transformer that consumes the controls is batched below.  The all-zero
        reactive branch is shared across the batch because it is deterministic.
        """
        if pixels.ndim != 5:
            raise ValueError("Expected [B,C,T,H,W] pixels")
        batch, channels, frames, height, width = pixels.shape
        if not 0 <= observed <= frames:
            raise ValueError("Observed frame count is outside the video")
        control_pixels = pixels.clone()
        control_pixels[:, :, observed:] = 0
        inactive = torch.stack(
            self.vae.encode([control_pixels[index] for index in range(batch)]), 0
        )
        reactive = self._zero_latent(pixels[0]).expand(batch, -1, -1, -1, -1)
        mask = torch.ones((frames, height, width), device=self.device)
        mask[:observed] = 0
        size = self.config.image_size
        mask = mask.view(frames, size // 8, 8, size // 8, 8).permute(2, 4, 0, 1, 3).flatten(0, 1)
        mask = F.interpolate(mask[None], size=inactive.shape[2:], mode="nearest-exact")[0]
        mask = mask.unsqueeze(0).expand(batch, -1, -1, -1, -1)
        return torch.cat((inactive, reactive, mask), 1).bfloat16(), inactive.shape[1:]

    def _forward(self, latent, control, text, timestep, capture=None):
        batch = latent.shape[0]
        timestep = torch.as_tensor(timestep, device=self.device).reshape(-1)
        if timestep.numel() == 1:
            timestep = timestep.expand(batch)
        elif timestep.numel() != batch:
            raise ValueError(f"Timestep batch {timestep.numel()} does not match latent batch {batch}")
        kwargs = {
            "hidden_states": latent.bfloat16(),
            "timestep": timestep,
            "encoder_hidden_states": text,
            "control_hidden_states": control,
            "control_hidden_states_scale": self.scales,
            "return_dict": False,
        }
        if capture is None:
            return self.model(**kwargs)[0]
        with capture:
            result = self.model(**kwargs)[0]
        if not torch.isfinite(result).all():
            raise RuntimeError("VACE teacher returned nonfinite velocity")
        return result

    def _readout(self, capture, anchors, content_mask, index=None):
        values = capture.result(index)
        if self.task in {"tracking", "shared"}:
            readout_values = capture.values if index is None else {
                key: value[index] for key, value in capture.values.items()
            }
            paths = multi_anchor_paths(readout_values, anchors, self.config, content_mask)
            if self.task == "tracking":
                return paths
            values.update(paths)
        return values

    @torch.no_grad()
    def extract_batch(self, teacher_rgbs, text_embeddings, text_masks, seeds, content_masks):
        """Extract targets for independent windows in one teacher micro-batch."""
        if isinstance(teacher_rgbs, torch.Tensor):
            rgb_batch = teacher_rgbs
        else:
            rgb_batch = torch.stack(list(teacher_rgbs), 0)
        if rgb_batch.ndim != 5:
            raise ValueError("Teacher batch must have shape [B,49,3,H,W]")
        count = rgb_batch.shape[0]
        if count < 1 or not all(len(x) == count for x in
                                (text_embeddings, text_masks, seeds, content_masks)):
            raise ValueError("extract_batch inputs must have equal, nonzero length")
        if rgb_batch.shape[1] != 49 or rgb_batch.shape[2] != 3 or rgb_batch.dtype != torch.uint8:
            raise ValueError("Teacher requires [B,49,3,H,W] uint8 videos")
        text = torch.stack([e.to(self.device, dtype=torch.bfloat16) for e in text_embeddings], 0)
        text_masks = [m.to(self.device) if torch.is_tensor(m) else torch.as_tensor(m, device=self.device)
                      for m in text_masks]

        # ---- observed: one batched capture with independent random latents ----
        past = rgb_batch[:, :33]
        warm = torch.cat((past[:, :1].expand(-1, 15, -1, -1, -1), past), 1)
        observed_pixels = self._pixels_batch(warm)
        observed_control, observed_shape = self._inputs_batch(observed_pixels, 48)
        observed_generators = [
            torch.Generator(device=self.device).manual_seed(int(seed)) for seed in seeds
        ]
        observed_latent = torch.stack([
            torch.randn(observed_shape, device=self.device, generator=generator)
            for generator in observed_generators
        ], 0)
        observed_capture = Capture(
            self.model, self.task, observed_shape[1], self.config.image_size // 16,
            torch.stack(text_masks, 0),
        )
        self._forward(
            observed_latent,
            observed_control,
            text,
            torch.tensor([1000.0], device=self.device),
            observed_capture,
        )
        def observed_readouts(capture):
            return [
                self._readout(capture, OBSERVED_ANCHORS, content_masks[index], index)
                for index in range(count)
            ]

        if _teacher_opt():
            # CPU beam decoding overlaps the NPU work of the future branch below.
            observed_future = self._readout_pool().submit(observed_readouts, observed_capture)
        else:
            observed_all = observed_readouts(observed_capture)
        del observed_capture, observed_latent, observed_control, observed_pixels, warm

        # ---- future: batched VAE/control, denoise, and final capture ----
        pixels = self._pixels_batch(rgb_batch)
        clean = torch.stack(
            self.vae.encode([pixels[index] for index in range(count)]), 0
        ).float()
        controls, _ = self._inputs_batch(pixels, 33)
        shape = clean.shape[1:]
        del pixels
        scheduler, begin, nodes = sparse_schedule(
            self.scheduler, self.config, device=self.device
        )
        timesteps = scheduler.timesteps[begin:]
        sigma = scheduler.sigmas[begin].float()
        noise = torch.stack([
            torch.randn(shape, device=self.device, generator=generator)
            for generator in observed_generators
        ], 0)
        latent = (1 - sigma) * clean + sigma * noise
        del clean, noise
        negative = self.negative.expand(count, *self.negative.shape[1:])
        parallel_cfg = _cfg_parallel()
        for index in range(self.config.denoise_updates):
            if parallel_cfg:
                pair = self._forward(
                    torch.cat((latent, latent), 0),
                    torch.cat((controls, controls), 0),
                    torch.cat((text, negative), 0),
                    timesteps[index])
                conditional, unconditional = pair[0:count], pair[count:]
                del pair
            else:
                conditional = self._forward(latent, controls, text, timesteps[index])
                unconditional = self._forward(latent, controls, negative, timesteps[index])
            velocity = unconditional.float() + self.config.guidance * (
                conditional.float() - unconditional.float())
            del conditional, unconditional
            latent = scheduler.step(velocity, timesteps[index], latent, return_dict=False)[0]
            del velocity
        if scheduler.step_index != begin + self.config.denoise_updates or (
            abs(float(scheduler.sigmas[scheduler.step_index]) - nodes[-1]["sigma"]) > 1e-7
        ):
            raise RuntimeError("Future denoising missed the requested readout endpoint")
        future_capture = Capture(
            self.model, self.task, shape[1], self.config.image_size // 16,
            torch.stack(text_masks, 0),
        )
        self._forward(
            latent,
            controls,
            text,
            timesteps[self.config.denoise_updates],
            future_capture,
        )
        future_all = [
            self._readout(future_capture, FUTURE_ANCHORS, content_masks[index], index)
            for index in range(count)
        ]
        self.protocol.update(
            actual_sigma=float(sigma),
            begin_index=begin,
            denoise_updates=self.config.denoise_updates,
            denoise_reference_updates=self.config.denoise_reference_updates,
            reference_endpoint_index=begin + self.config.denoise_reference_updates,
            denoise_schedule=nodes,
            capture_sigma=nodes[-1]["sigma"],
            capture_timestep=float(timesteps[self.config.denoise_updates]),
        )
        del latent, controls, future_capture, negative, text
        if _teacher_opt():
            observed_all = observed_future.result()
        targets = []
        for i in range(count):
            target = {"observed": observed_all[i], "future": future_all[i]}
            validate_target(target, self.task, grid=self.config.image_size // 16,
                            text_length=len(text_masks[i]))
            targets.append(target)
        return targets

    @torch.no_grad()
    def extract(self, teacher_rgb, text_embedding, text_mask, seed, content_mask):
        return self.extract_batch(
            [teacher_rgb], [text_embedding], [text_mask], [seed], [content_mask]
        )[0]

    def close(self):
        del self.model, self.vae, self.negative, self.scales


def content_token_masks(text_root, prompts):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(Path(text_root) / "tokenizer", local_files_only=True)
    stop = {"the", "a", "an", "and", "to", "of", "on", "in", "at", "is", "it", "up"}
    result = {}
    for prompt in prompts:
        tokens = tokenizer(prompt, padding="max_length", max_length=512, truncation=True)
        pieces = tokenizer.convert_ids_to_tokens(tokens["input_ids"])
        result[prompt] = np.array(
            [
                bool(valid)
                and token not in tokenizer.all_special_tokens
                and token.lstrip(chr(9601)).lower() not in stop
                for token, valid in zip(pieces, tokens["attention_mask"], strict=True)
            ]
        )
    return result
