#!/usr/bin/env python3
"""Cosmos-Predict2.5 DiT on one Ascend NPU: weight loading, denoising accuracy, speed.

Accuracy (real LIBERO clips, OpenWAM canvas, Wan2.1 VAE latents, empty-prompt context):
noise the clean latents to sigma, predict the velocity, reconstruct ``x0 = x_t - sigma * v``
and report the relative error of the predicted (non-conditioning) latent frames against two
baselines (``x_t`` itself, and repeating the clean first frame).  Variants: the torch shim
RoPE (correct convention), a deliberately wrong RoPE (interleaved pairs) and no RoPE as
negative controls, and torch_npu fused RMSNorm / RoPE.

Speed (bf16, B=4, 9 frames at 384x320 -> 3 x 24 x 20 tokens, 512 text tokens): DiT forward and
forward+backward with the shim vs the fused kernels; joint-attention SDPA at the StructWAM
sequence length with no / bool / float masks, and ``npu_fusion_attention`` with an additive
bias (``pse``) as a fused alternative for the routing bias.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401
from torch_npu.contrib import transfer_to_npu  # noqa: F401

W = Path("/home/ma-user/work/2337888765/workspace")
RESULTS = {}


def log(name, value):
    RESULTS[name] = value
    print(name, json.dumps(value) if not isinstance(value, str) else value, flush=True)


def sync():
    torch.npu.synchronize()


def timed(fn, warmup=2, iters=5):
    for _ in range(warmup):
        fn()
    sync()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - start) / iters


# ----------------------------------------------------------------------------- fused kernels

def fused_rms_forward(self, x):
    if self.zero_centered_gamma:
        raise NotImplementedError
    return torch_npu.npu_rms_norm(x, self.weight.to(x.dtype), epsilon=self.eps)[0]


def fused_rope(t, freqs, tensor_format="sbhd", interleaved=False, **_):
    length = t.shape[1] if tensor_format == "bshd" else t.shape[0]
    freqs = freqs[:length]
    if tensor_format == "bshd":
        freqs = freqs.transpose(0, 1)
    cos, sin = torch.cos(freqs).to(t.dtype), torch.sin(freqs).to(t.dtype)
    return torch_npu.npu_rotary_mul(t, cos, sin)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=W / "models/Cosmos-Predict2.5-2B")
    parser.add_argument("--variant", default="robot/action-cond")
    parser.add_argument("--clips", type=int, default=4)
    parser.add_argument("--output", type=Path, default=W / "runs/cosmos_npu_check_20261020/results.json")
    parser.add_argument("--timestep-scale", type=float, default=None,
                        help="override the DiT timestep_scale (upstream 2B configs use 0.001)")
    parser.add_argument("--fp32-strategy", action="store_true",
                        help="upstream use_wan_fp32_strategy via float32 islands (npu_compat.apply_fp32_strategy)")
    parser.add_argument("--skip-speed", action="store_true")
    args = parser.parse_args()
    device = torch.device("npu", 0)
    torch.npu.set_device(device)

    from openwam.model.video_backbone.cosmos_predict25.npu_compat import ensure_transformer_engine

    log("transformer_engine_mode", ensure_transformer_engine())
    import transformer_engine as te
    import cosmos_predict2._src.predict2.networks.minimal_v4_dit as v4
    from cosmos_predict2._src.predict2.networks.minimal_v1_lvg_dit import MinimalV1LVGDiT
    from transformer_engine.pytorch.attention import rope as shim_rope

    from openwam.model.video_backbone.cosmos_predict25 import dit_forward as df
    from openwam.model.video_backbone.cosmos_predict25._vae_utils import _pil_video_to_tensor
    from openwam.model.video_backbone.cosmos_predict25.pipeline_builder import (
        _COSMOS25_2B_NET_KWARGS, _build_cosmos_predict25_vae, _load_state_dict_into_net, _resolve_checkpoint_path)

    log("transformer_engine", getattr(te, "__version__", "?"))

    # ------------------------------------------------------------------ load
    tick = time.time()
    overrides = {"atten_backend": "torch"}
    if args.timestep_scale is not None:
        overrides["timestep_scale"] = args.timestep_scale
    log("net_overrides", {**overrides, "fp32_strategy": args.fp32_strategy,
                           "timestep_scale_default": _COSMOS25_2B_NET_KWARGS.get("timestep_scale"),
                           "te_shim_fused_env": __import__("os").environ.get("TE_SHIM_FUSED", "1")})
    net = MinimalV1LVGDiT(sac_config=v4.SACConfig(mode=v4.CheckpointMode.NONE),
                          **{**_COSMOS25_2B_NET_KWARGS, **overrides})
    checkpoint = _resolve_checkpoint_path(args.model, args.variant)
    missing, unexpected = _load_state_dict_into_net(net, checkpoint)
    net = net.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    if args.fp32_strategy:
        from openwam.model.video_backbone.cosmos_predict25.npu_compat import apply_fp32_strategy

        log("fp32_islands", apply_fp32_strategy(net))
    log("load", {"checkpoint": checkpoint.name, "seconds": round(time.time() - tick, 1), "missing": len(missing),
                 "missing_head": missing[:5], "unexpected": len(unexpected), "unexpected_head": unexpected[:5],
                 "params_b": round(sum(p.numel() for p in net.parameters()) / 1e9, 3)})
    vae = _build_cosmos_predict25_vae(args.model / "tokenizer.pth", device=device, dtype=torch.bfloat16)
    empty = torch.load(args.model / args.variant / "cr1_empty_string_text_embeddings.pt", map_location="cpu")
    context1 = net.crossattn_proj(empty.to(device, torch.bfloat16))                      # [1, 512, 1024]

    # ------------------------------------------------------------------ data
    from omegaconf import OmegaConf

    from openwam.dataloader.registry import build_dataset

    dataset = build_dataset(OmegaConf.load("configs/dataloader/libero.yaml"), split="train")
    picks = np.random.default_rng(0).choice(len(dataset), args.clips, replace=False)
    clips = [dataset[int(i)]["video"] for i in picks]
    with torch.no_grad():
        video = _pil_video_to_tensor(clips).to(device, torch.bfloat16)
        latents = vae.encode(video).to(torch.bfloat16)
    log("latents", {"shape": list(latents.shape), "std": round(float(latents.float().std()), 4),
                    "episodes": [int(i) for i in picks]})
    batch, _, frames, height, width = latents.shape
    context = context1.expand(batch, -1, -1)
    condition = torch.zeros(batch, 1, frames, height, width, dtype=torch.bfloat16, device=device)
    condition[:, :, 0] = 1

    def velocity(x_t, sigma, grad=False):
        with torch.set_grad_enabled(grad):
            state = df.prepare_block_loop(net, latents=x_t, context=context,
                                          timestep=torch.full((x_t.shape[0],), sigma * 1000.0, device=device,
                                                              dtype=torch.bfloat16), condition_mask=condition)
            for block in range(len(net.blocks)):
                state = df.run_block(net, block, state)
            return df.finalize_block_loop(net, state)

    # ------------------------------------------------------------------ accuracy
    x0 = latents.float()
    future = x0[:, :, 1:]
    scale = ((future - future.mean()) ** 2).mean()
    rope_variants = {
        "shim_rope": shim_rope.apply_rotary_pos_emb,
        "wrong_rope_interleaved": lambda t, f, **kw: shim_rope.apply_rotary_pos_emb(t, f, interleaved=True,
                                                                                   tensor_format=kw.get("tensor_format", "sbhd")),
        "no_rope": lambda t, f, **kw: t,
    }
    original_rope, original_rms = v4.apply_rotary_pos_emb, te.pytorch.RMSNorm.forward
    accuracy = {}
    reference = {}
    for name, rope in list(rope_variants.items()) + [("fused_rms_rope", None)]:
        if rope is None:
            v4.apply_rotary_pos_emb, te.pytorch.RMSNorm.forward = fused_rope, fused_rms_forward
        else:
            v4.apply_rotary_pos_emb, te.pytorch.RMSNorm.forward = rope, original_rms
        rows = {}
        try:
            for sigma in (0.3, 0.6, 0.9):
                errors, identity, copy_first, diffs = [], [], [], []
                for seed in (0, 1):
                    noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(seed)).to(device)
                    x_t = (1 - sigma) * x0 + sigma * noise
                    x_t[:, :, 0] = x0[:, :, 0]
                    v = velocity(x_t.to(torch.bfloat16), sigma).float()
                    x0_hat = x_t - sigma * v
                    errors.append(float(((x0_hat[:, :, 1:] - future) ** 2).mean() / scale))
                    identity.append(float(((x_t[:, :, 1:] - future) ** 2).mean() / scale))
                    copy_first.append(float(((x0[:, :, :1] - future) ** 2).mean() / scale))
                    if name == "shim_rope":
                        reference[sigma, seed] = v
                    elif name == "fused_rms_rope":
                        diffs.append(float((v - reference[sigma, seed]).abs().max()))
                rows[f"sigma{sigma}"] = {"rel_err": round(float(np.mean(errors)), 4),
                                         "baseline_noisy_input": round(float(np.mean(identity)), 4),
                                         "baseline_copy_first_frame": round(float(np.mean(copy_first)), 4)}
                if diffs:
                    rows[f"sigma{sigma}"]["max_abs_diff_vs_shim_velocity"] = round(max(diffs), 5)
        except Exception as exc:  # report and continue with the other variants
            rows = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        accuracy[name] = rows
        log(f"accuracy/{name}", rows)
    v4.apply_rotary_pos_emb, te.pytorch.RMSNorm.forward = original_rope, original_rms

    if args.skip_speed:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(RESULTS, indent=2) + "\n")
        print("wrote", args.output)
        return

    # ------------------------------------------------------------------ speed: DiT
    noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(2)).to(device)
    x_t = ((1 - 0.6) * x0 + 0.6 * noise).to(torch.bfloat16)
    target = (noise - x0).to(torch.bfloat16)
    speed = {}
    for name, patch in (("shim", (original_rope, original_rms)), ("fused", (fused_rope, fused_rms_forward))):
        v4.apply_rotary_pos_emb, te.pytorch.RMSNorm.forward = patch
        try:
            forward = timed(lambda: velocity(x_t, 0.6))
            net.requires_grad_(True)

            def step():
                net.zero_grad(set_to_none=True)
                loss = ((velocity(x_t, 0.6, grad=True).float() - target.float()) ** 2).mean()
                loss.backward()

            torch.npu.reset_peak_memory_stats()
            train = timed(step, warmup=1, iters=3)
            peak = torch.npu.max_memory_allocated() / 2**30
            net.requires_grad_(False)
            net.zero_grad(set_to_none=True)
            speed[name] = {"forward_s": round(forward, 3), "forward_backward_s": round(train, 3),
                           "peak_gib": round(peak, 1)}
        except Exception as exc:
            net.requires_grad_(False)
            speed[name] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        log(f"speed/dit_{name}", speed[name])
    v4.apply_rotary_pos_emb, te.pytorch.RMSNorm.forward = original_rope, original_rms

    # ------------------------------------------------------------------ speed: joint attention
    heads, dim = 16, 128
    s_video, s_action, s_structure = frames * (height // 2) * (width // 2), 33, 234
    total = s_video + s_action + s_structure
    q, k, v = (torch.randn(batch, heads, total, dim, device=device, dtype=torch.bfloat16, requires_grad=True)
               for _ in range(3))
    allowed = torch.ones(total, total, dtype=torch.bool, device=device)
    allowed[:s_video, s_video:s_video + s_action] = False                     # e.g. video does not read action
    allowed[s_video + s_action:, s_video:s_video + s_action] = False          # structure never reads action
    floor = torch.finfo(torch.bfloat16).min / 2
    routing = torch.zeros(batch, 1, total, total, device=device, dtype=torch.bfloat16)
    routing[:, :, s_video:s_video + s_action, s_video + s_action:] = 1.0      # routing bias on action rows
    bias = routing.masked_fill(~allowed, floor)                                # float mask = mask + routing bias
    sdpa = torch.nn.functional.scaled_dot_product_attention

    def run(fn):
        def go():
            fn().float().sum().backward()
            for t in (q, k, v):
                t.grad = None
        return go

    with torch.no_grad():
        reference_bool = sdpa(q, k, v, attn_mask=allowed).float()
        reference_float = sdpa(q, k, v, attn_mask=bias).float()
    cases = {
        "sdpa_no_mask": (lambda: sdpa(q, k, v), None),
        "sdpa_bool_mask_SxS": (lambda: sdpa(q, k, v, attn_mask=allowed), None),
        "sdpa_bool_mask_Bx1xSxS": (lambda: sdpa(q, k, v, attn_mask=allowed.expand(batch, 1, total, total)), reference_bool),
        "sdpa_float_mask_Bx1xSxS": (lambda: sdpa(q, k, v, attn_mask=bias), None),
    }
    s_rows = (s_video, s_video + s_action)
    cases["grouped_rows_bool_plus_action_float (StructWAM driver)"] = (lambda: torch.cat((
        sdpa(q[:, :, :s_rows[0]], k, v, attn_mask=allowed[:s_rows[0]]),
        sdpa(q[:, :, s_rows[0]:s_rows[1]], k, v, attn_mask=bias[:, :, s_rows[0]:s_rows[1]]),
        sdpa(q[:, :, s_rows[1]:], k, v, attn_mask=allowed[s_rows[1]:])), 2), reference_float)
    if hasattr(torch_npu, "npu_fusion_attention"):
        masked = ~allowed                                                      # npu_fusion_attention: True = masked out
        pse = routing.expand(batch, heads, total, total).contiguous()
        cases["npu_fusion_attention_bool_mask"] = (lambda: torch_npu.npu_fusion_attention(
            q, k, v, heads, "BNSD", atten_mask=masked, scale=dim ** -0.5)[0], reference_bool)
        cases["npu_fusion_attention_bool_mask_plus_pse_bias"] = (lambda: torch_npu.npu_fusion_attention(
            q, k, v, heads, "BNSD", pse=pse, atten_mask=masked, scale=dim ** -0.5)[0], reference_float)
    attention = {"sequence": total}
    for name, (fn, reference) in cases.items():
        try:
            entry = {"forward_backward_ms": round(timed(run(fn)) * 1000, 2)}
            if reference is not None:
                with torch.no_grad():
                    entry["max_abs_diff_vs_sdpa"] = round(float((fn().float() - reference).abs().max()), 4)
            attention[name] = entry
        except Exception as exc:
            attention[name] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        log(f"speed/attention/{name}", attention[name])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(RESULTS, indent=2) + "\n")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
