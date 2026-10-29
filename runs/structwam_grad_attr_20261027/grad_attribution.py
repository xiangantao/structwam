"""Which loss terms train the StructWAM structure side?  Gradient attribution on a checkpoint.

For training batches (libero_struct reader, the checkpoint's own config, training mode so the gate
teacher forcing and timestep / noise sampling match training) one forward is run and the gradient
of every *weighted* loss term is taken separately on the structure-side parameters:

    video   lambda_video * structure-weighted flow-matching video loss
    action  lambda_action * action flow-matching loss
    phase / align / track   the StructWAM auxiliaries with their lambdas (sketch has lambda 0)

A second forward with the same seed and ``video_detach = {structure}`` gives the video gradient
that reaches the structure side only *indirectly* (video reads action, action reads structure);
direct = video - indirect.

Per parameter group, over all batches of all ranks:
  * projection share  <G_k, G_total> / |G_total|^2   (G = mean gradient; the shares sum to 1)
  * |G_k|, mean per-batch |g_k|, and cosine(G_k, G_action)

    torchrun --nproc_per_node=8 grad_attribution.py --checkpoint CKPT --output DIR --batches 6
"""

import argparse
import copy
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch_npu  # noqa: F401
from torch_npu.contrib import transfer_to_npu  # noqa: F401

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from openwam.dataloader import build_dataset
from openwam.deploy.model_loader import load_from_checkpoint_dir
from openwam.model.architectures.tri_system import struct_system
from openwam.structure import losses as structure_losses

TERMS = ("video", "action", "phase", "align", "track")
MODULES = ("structure_expert", "phase_gate", "routing", "align_projector")

# ----------------------------------------------------------------------------- capture loss terms

CAPTURE = {}
_video_loss = struct_system.TriSystemStructArchitecture._compute_video_loss
_action_loss = struct_system.TriSystemStructArchitecture._compute_action_loss


def _capture_video(self, *args, **kwargs):
    CAPTURE["video"] = _video_loss(self, *args, **kwargs)
    return CAPTURE["video"]


def _capture_action(self, *args, **kwargs):
    CAPTURE["action"] = _action_loss(self, *args, **kwargs)
    return CAPTURE["action"]


struct_system.TriSystemStructArchitecture._compute_video_loss = _capture_video
struct_system.TriSystemStructArchitecture._compute_action_loss = _capture_action


def weighted_terms(arch, result, lambda_video, lambda_action):
    """Non-detached weighted terms, recomputed exactly as compute_loss does; checked against the total."""
    aux, cfg = arch._aux, arch.loss_cfg
    terms = {"video": lambda_video * CAPTURE["video"], "action": lambda_action * CAPTURE["action"]}
    if aux.get("phase_target") is not None:
        terms["phase"] = F.binary_cross_entropy_with_logits(aux["gate_logit"].float(), aux["phase_target"].float())
    if aux.get("hidden") is not None:
        hidden = structure_losses.hidden_grid(aux["hidden"], *aux["grid"])
        terms["align"] = structure_losses.alignment_loss(hidden, arch.align_projector, aux["structure"], arch.layout)
    if aux.get("track") is not None:
        terms["track"] = aux["track"]
    for name in ("phase", "align", "track"):
        if name in terms:
            weight = float(cfg.get(f"lambda_{name}", 0.0))
            terms[name] = weight * terms[name] if weight > 0 else None
    terms = {k: v for k, v in terms.items() if v is not None}
    total = sum(terms.values())
    gap = abs(float(total) - float(result["loss"]))
    if gap > 1e-3 * max(1.0, abs(float(result["loss"]))):
        raise RuntimeError(f"recomputed terms {float(total):.6f} != total loss {float(result['loss']):.6f}")
    return terms


# ----------------------------------------------------------------------------- parameter groups


def group_of(name: str, num_layers: int) -> str:
    top = name.split(".")[0]
    if top != "structure_expert":
        return top
    parts = name.split(".")
    if parts[1] == "blocks":
        layer = int(parts[2])
        third = ("early", "mid", "late")[min(2, 3 * layer // num_layers)]
        kind = "qkv" if any(t in name for t in (".q.", ".k.", ".v.", "qkv", "to_q", "to_k", "to_v")) else "other"
        return f"structure_expert.blocks_{third}.{kind}"
    return f"structure_expert.{parts[1]}"


def flat_grads(term, params):
    grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).float().reshape(-1)
                      for g, p in zip(grads, params)])


# ----------------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batches", type=int, default=6, help="batches per rank")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--no-indirect", action="store_true", help="skip the video_detach=[structure] pass")
    args = parser.parse_args()

    dist.init_process_group("hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    out = Path(args.output)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.checkpoint)
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(ckpt.parent / "config.yaml")
    ds = build_dataset(cfg.dataloader, split="train")
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(ds), size=world * args.batches * args.batch_size, replace=False)
    mine = indices.reshape(world, args.batches, args.batch_size)[rank]

    _, arch = load_from_checkpoint_dir(str(ckpt.parent), device=f"cuda:{local}", ckpt_name=ckpt.name)
    arch.normalizer = None
    arch.init_training_schedulers(1000)                       # as OpenWAMTrainer does
    arch.train()
    arch.requires_grad_(False)
    named = [(n, p) for n, p in arch.named_parameters() if n.split(".")[0] in MODULES]
    for _, p in named:
        p.requires_grad_(True)
    params = [p for _, p in named]
    num_layers = arch.video_backbone.num_layers
    groups = defaultdict(list)
    offset = 0
    for name, p in named:
        groups[group_of(name, num_layers)].append((offset, offset + p.numel()))
        offset += p.numel()
    group_index = {g: torch.cat([torch.arange(a, b) for a, b in spans]) for g, spans in groups.items()}
    if rank == 0:
        sizes = {g: int(len(i)) for g, i in group_index.items()}
        print("param groups", json.dumps(sizes, indent=1), "total", offset, flush=True)

    t = cfg.training
    lambda_video, lambda_action = float(t.lambda_video), float(t.lambda_action)
    keys = list(TERMS) + ([] if args.no_indirect else ["video_indirect"])
    sums = {k: torch.zeros(offset, device=f"cuda:{local}") for k in keys}
    batch_norms = defaultdict(lambda: defaultdict(list))   # term -> group -> per-batch norm
    batch_cos = defaultdict(lambda: defaultdict(list))     # term -> group -> per-batch cos with action
    log = []
    driver = arch._driver
    for b, batch_idx in enumerate(mine):
        started = time.monotonic()
        samples = [ds[int(i)] for i in batch_idx]
        with torch.no_grad():
            inputs = arch.prepare_inputs(samples)
        inputs["use_gradient_checkpointing"] = bool(t.get("use_gradient_checkpointing", True))
        step_seed = args.seed * 1000 + rank * 100 + b
        grads, values = {}, {}
        passes = [("normal", frozenset())] + ([] if args.no_indirect else [("indirect", frozenset({"structure"}))])
        for tag, detach in passes:
            driver.video_detach = detach
            torch.manual_seed(step_seed)
            torch.cuda.manual_seed_all(step_seed)
            CAPTURE.clear()
            result = arch.compute_loss(**copy.copy(inputs), lambda_video=lambda_video, lambda_action=lambda_action)
            terms = weighted_terms(arch, result, lambda_video, lambda_action)
            if tag == "normal":
                for k, v in terms.items():
                    grads[k] = flat_grads(v, params)
                    values[k] = float(v)
            else:
                grads["video_indirect"] = flat_grads(terms["video"], params)
                values["video_indirect_loss_check"] = float(terms["video"])
            del result, terms
        driver.video_detach = frozenset()
        for k, g in grads.items():
            sums[k] += g
            for name, idx in group_index.items():
                gi = g[idx.to(g.device)]
                batch_norms[k][name].append(float(gi.norm()))
                if "action" in grads:
                    ga = grads["action"][idx.to(g.device)]
                    batch_cos[k][name].append(float(F.cosine_similarity(gi, ga, dim=0)))
        entry = {"rank": rank, "batch": b, "seconds": round(time.monotonic() - started, 1), "losses": values,
                 "struct_norm": {k: float(g.norm()) for k, g in grads.items()}}
        log.append(entry)
        print("BATCH", json.dumps(entry), flush=True)
        del grads

    # ----- reduce over ranks
    n_batches = world * args.batches
    for k in keys:
        dist.all_reduce(sums[k])
        sums[k] /= n_batches
    gathered = [None] * world
    dist.all_gather_object(gathered, {"norms": {k: dict(v) for k, v in batch_norms.items()},
                                      "cos": {k: dict(v) for k, v in batch_cos.items()}, "log": log})
    if rank == 0:
        norms, coss, logs = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list)), []
        for part in gathered:
            logs += part["log"]
            for k, d in part["norms"].items():
                for g, v in d.items():
                    norms[k][g] += v
            for k, d in part["cos"].items():
                for g, v in d.items():
                    coss[k][g] += v
        total = sum(sums[k] for k in TERMS)
        report = {"checkpoint": str(ckpt), "batches": n_batches, "batch_size": args.batch_size,
                  "video_detach_in_config": list(cfg.model.architecture.structure.get("video_detach", [])),
                  "lambdas": {"video": lambda_video, "action": lambda_action,
                              **{k: float(arch.loss_cfg.get(f"lambda_{k}", 0.0)) for k in ("phase", "align", "track")}},
                  "groups": {}}
        for name, idx in list(group_index.items()) + [("ALL", torch.arange(offset))]:
            idx = idx.to(total.device)
            gt = total[idx]
            denom = float(gt.dot(gt)) or 1.0
            row = {"params": int(len(idx)), "total_mean_grad_norm": float(gt.norm()), "terms": {}}
            for k in keys:
                gk = sums[k][idx]
                row["terms"][k] = {
                    "projection_share": float(gk.dot(gt)) / denom,
                    "mean_grad_norm": float(gk.norm()),
                    "per_batch_norm": float(np.mean(norms[k][name])) if name in norms[k] else None,
                    "cos_with_action_mean_grad": float(F.cosine_similarity(gk, sums["action"][idx], dim=0)),
                    "cos_with_total_mean_grad": float(F.cosine_similarity(gk, gt, dim=0)),
                    "per_batch_cos_with_action": float(np.mean(coss[k][name])) if name in coss[k] else None,
                }
            if "video_indirect" in sums:
                direct = sums["video"][idx] - sums["video_indirect"][idx]
                row["terms"]["video_direct"] = {"projection_share": float(direct.dot(gt)) / denom,
                                                "mean_grad_norm": float(direct.norm())}
            report["groups"][name] = row
        (out / "summary.json").write_text(json.dumps(report, indent=2))
        (out / "batches.jsonl").write_text("".join(json.dumps(e) + "\n" for e in logs))
        print("COMPLETE", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
