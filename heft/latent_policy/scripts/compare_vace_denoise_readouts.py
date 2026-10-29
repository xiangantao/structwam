#!/usr/bin/env python3
"""Compare teacher readouts after 0..13 updates and directly re-noised GT.

All errors are disagreements with the step-13 readout, not errors against GT.
This diagnostic explicitly retains the 13-update reference and does not train a student.
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from vace_latent_distill.config import FUTURE_ANCHORS, NEGATIVE_PROMPT, PipelineConfig
from vace_latent_distill.data import decode_indices, stable_seed
from vace_latent_distill.teacher import Capture, VaceTeacher, content_token_masks, multi_anchor_paths


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def progress(**values):
    print(json.dumps(values, allow_nan=False), flush=True)


def correlation(a, b, ranks=False):
    a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    if ranks:
        from scipy.stats import rankdata

        a, b = rankdata(a), rankdata(b)
    a, b = a - a.mean(), b - b.mean()
    scale = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / scale) if scale > 1e-10 else None


def valid_mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def feature_metrics(value, reference, active):
    value, reference = value[9:].float(), reference[9:].float()
    similarity = F.cosine_similarity(value, reference, dim=-1)
    centered = value - value.mean(1, keepdim=True)
    ref_centered = reference - reference.mean(1, keepdim=True)
    return {
        "cosine": float(similarity.mean()),
        "cosine_p10": float(torch.quantile(similarity, 0.1)),
        "cosine_changed_pixels": float(similarity[active].mean()) if active.any() else None,
        "spatial_centered_cosine": float(F.cosine_similarity(centered, ref_centered, dim=-1).mean()),
        "relative_l2": float((value - reference).norm() / reference.norm().clamp_min(1e-8)),
    }


def spatial_affinity(value, reference):
    rng = np.random.default_rng(42)
    cells = value.shape[1]
    a, b = rng.integers(cells, size=(2, 2048))
    left = (F.normalize(value[9:].float(), dim=-1)[:, a]
            * F.normalize(value[9:].float(), dim=-1)[:, b]).sum(-1)
    right = (F.normalize(reference[9:].float(), dim=-1)[:, a]
             * F.normalize(reference[9:].float(), dim=-1)[:, b]).sum(-1)
    return valid_mean([correlation(x.numpy(), y.numpy()) for x, y in zip(left, right)])


def relation_metrics(values, reference, energy, temperature, cell_px):
    result = {}
    cells = values["track_q"].shape[1]
    grid = round(cells**0.5)
    for group in ("all", "changed_pixels"):
        rows = []
        for source in range(8, 12):
            if group == "all":
                ids = torch.linspace(0, cells - 1, 64).long()
            else:
                ids = torch.where(energy[source - 8] > 5)[0]
                ids = ids[energy[source - 8, ids].argsort(descending=True)[:64]]
                if not len(ids):
                    continue
            logits = values["track_q"][source, ids].float() @ values["track_k"][source + 1].float().T
            truth = reference["track_q"][source, ids].float() @ reference["track_k"][source + 1].float().T
            p = (logits / temperature).log_softmax(-1)
            r = (truth / temperature).softmax(-1)
            prediction, target = logits.argmax(-1), truth.argmax(-1)
            delta = torch.stack((prediction % grid - target % grid,
                                 prediction // grid - target // grid), -1).float().norm(dim=-1) * cell_px
            rows.append({
                "logit_correlation": correlation(logits.numpy(), truth.numpy()),
                "kl_to_step13": float(F.kl_div(p, r, reduction="none").sum(-1).mean()),
                "argmax_agreement": float((prediction == target).float().mean()),
                "argmax_delta_px": float(delta.mean()),
                "argmax_within8px": float((delta <= 8).float().mean()),
            })
        result[group] = {
            name: valid_mean([row[name] for row in rows])
            for name in ("logit_correlation", "kl_to_step13", "argmax_agreement",
                         "argmax_delta_px", "argmax_within8px")
        }
    return result


def fixed_paths(values, seeds, config):
    import demo_main_prompt_k_tracking as demo

    grid = config.image_size // 16
    decoded = []
    k_values = {22: values["k22"].numpy(), 15: values["k15"].numpy()}
    for anchor, chosen in zip(FUTURE_ANCHORS, seeds, strict=True):
        end = min(anchor + 3, values["track_q"].shape[0])
        region = demo.k_region_over_time(k_values, chosen, anchor, grid)
        path = demo.track_dense(
            values["track_q"][:end].numpy().reshape(-1, 128),
            values["track_k"][:end].numpy().reshape(-1, 128), chosen, anchor, grid, region[:end],
            topk=config.beam_topk, beam_size=config.beam_size, max_step=config.beam_max_step,
            motion_weight=config.motion_weight, local_weight=config.local_weight,
            region_weight=config.region_weight, acceleration_weight=config.acceleration_weight,
            descriptor_ema=config.descriptor_ema, direction="q_to_k",
        )
        path = demo.regularize_paths(path, config.regularize_max_step,
                                     config.regularize_max_acceleration, passes=2)
        decoded.append(path[:, 1:])
    return np.stack(decoded)


def describe(values, reference, energy, content, seeds, reference_paths, config, temperature, cell_px):
    active = energy > 5
    output = {"changed_pixel_fraction": float(active.float().mean())}
    for name in ("k22", "k15", "track_q", "track_k"):
        output[name] = feature_metrics(values[name], reference[name], active)
        if name.startswith("k"):
            output[name]["within_frame_affinity_correlation"] = spatial_affinity(values[name], reference[name])
    attention, truth = values["text_attention"][9:].float(), reference["text_attention"][9:].float()
    middle = (attention + truth) / 2
    js = 0.5 * ((attention * (attention.clamp_min(1e-8).log() - middle.clamp_min(1e-8).log())).sum(-1)
                + (truth * (truth.clamp_min(1e-8).log() - middle.clamp_min(1e-8).log())).sum(-1))
    heat, reference_heat = attention[..., content].max(-1).values, truth[..., content].max(-1).values
    topk = max(1, round(heat.shape[1] * 0.05))
    overlap = []
    for a, b in zip(heat, reference_heat):
        aa, bb = set(a.topk(topk).indices.tolist()), set(b.topk(topk).indices.tolist())
        overlap.append(len(aa & bb) / len(aa | bb))
    output["semantic"] = {
        "word_distribution_js": float(js.mean()),
        "content_heat_correlation": valid_mean([correlation(a.numpy(), b.numpy())
                                                for a, b in zip(heat, reference_heat)]),
        "top5pct_heat_iou": float(np.mean(overlap)),
    }
    depth, ref_depth = -values["k16"][9:, :, 52], -reference["k16"][9:, :, 52]
    output["depth"] = {
        "spearman": valid_mean([correlation(a.numpy(), b.numpy(), ranks=True) for a, b in zip(depth, ref_depth)]),
        "mae_over_reference_spatial_std": float((depth - ref_depth).abs().mean()
                                                / ref_depth.std(dim=1).mean().clamp_min(1e-8)),
    }
    output["qk_relation"] = relation_metrics(values, reference, energy, temperature, cell_px)
    paths = fixed_paths(values, seeds, config)
    delta = np.linalg.norm(paths - reference_paths, axis=-1) * cell_px
    seed_xy = np.stack((seeds % (config.image_size // 16), seeds // (config.image_size // 16)), -1)
    moving = np.linalg.norm(reference_paths - seed_xy[:, :, None], axis=-1) * cell_px > 8
    output["beam_paths"] = {
        "delta_to_step13_px": float(delta.mean()),
        "within8px_of_step13": float((delta <= 8).mean()),
        "reference_moving_point_fraction": float(moving.mean()),
        "delta_reference_moving_px": float(delta[moving].mean()) if moving.any() else None,
    }
    return output


@torch.no_grad()
def trace(teacher, items, maximum, direct_steps):
    rgb = torch.stack([item["teacher_rgb"] for item in items])
    batch = len(items)
    text = torch.stack([item["text"] for item in items]).to(teacher.device, dtype=torch.bfloat16)
    masks = torch.stack([item["text_mask"] for item in items]).to(teacher.device)
    generators = [torch.Generator(device=teacher.device).manual_seed(item["metadata"]["teacher_seed"])
                  for item in items]
    pixels = teacher._pixels_batch(rgb)
    clean = torch.stack(teacher.vae.encode(list(pixels)), 0).float()
    control, _ = teacher._inputs_batch(pixels, 33)
    del pixels
    # Consume the same observed-branch RNG draws as online extract_batch().
    for generator in generators:
        torch.randn((clean.shape[1], 12, *clean.shape[-2:]), device=teacher.device, generator=generator)
    noise = torch.stack([torch.randn(clean.shape[1:], device=teacher.device, generator=generator)
                         for generator in generators])
    scheduler = teacher.scheduler
    scheduler.set_timesteps(teacher.config.scheduler_steps, device=teacher.device)
    begin = int((scheduler.sigmas[:-1] - teacher.config.sigma).abs().argmin())
    scheduler.set_begin_index(begin)
    timesteps = scheduler.timesteps[begin:]
    if maximum >= len(timesteps):
        raise ValueError("The trace endpoint must still have a capture timestep")
    latent = (1 - scheduler.sigmas[begin]) * clean + scheduler.sigmas[begin] * noise
    pair_control = torch.cat((control, control), 0)
    pair_text = torch.cat((text, teacher.negative.expand(batch, -1, -1)), 0)
    snapshots, schedule = [], []
    started = time.monotonic()
    for step in range(maximum + 1):
        update = step < maximum
        capture = Capture(teacher.model, "shared", clean.shape[2], teacher.config.image_size // 16,
                          torch.cat((masks, masks), 0) if update else masks)
        velocity = teacher._forward(torch.cat((latent, latent), 0) if update else latent,
                                    pair_control if update else control,
                                    pair_text if update else text, timesteps[step], capture)
        snapshots.append({name: value[:batch].clone() for name, value in capture.values.items()})
        schedule.append({"step": step, "sigma": float(scheduler.sigmas[begin + step]),
                         "timestep": float(timesteps[step]), "elapsed_seconds": time.monotonic() - started})
        progress(stage="captured", samples=[item["metadata"]["id"] for item in items], **schedule[-1])
        if update:
            conditional, unconditional = velocity.chunk(2)
            guided = unconditional.float() + teacher.config.guidance * (conditional.float() - unconditional.float())
            latent = scheduler.step(guided, timesteps[step], latent, return_dict=False)[0]
            del conditional, unconditional, guided
        del velocity, capture
    direct = {}
    for step in direct_steps:
        sigma = scheduler.sigmas[begin + step]
        capture = Capture(teacher.model, "shared", clean.shape[2], teacher.config.image_size // 16, masks)
        teacher._forward((1 - sigma) * clean + sigma * noise, control, text, timesteps[step], capture)
        direct[step] = {name: value.clone() for name, value in capture.values.items()}
        progress(stage="direct_noise_captured", step=step, sigma=float(sigma))
    return snapshots, direct, schedule


def flatten(values, prefix=""):
    result = {}
    for name, value in values.items():
        key = f"{prefix}/{name}" if prefix else name
        result.update(flatten(value, key) if isinstance(value, dict) else {key: value})
    return result


def summarize(records):
    groups = {}
    for row in records:
        groups.setdefault((row["mode"], row["step"]), []).append(flatten(row["metrics"]))
    return [{"mode": mode, "step": step, "samples": len(rows),
             "metrics": {name: valid_mean([row[name] for row in rows]) for name in rows[0]}}
            for (mode, step), rows in sorted(groups.items())]


def plot_summary(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = [row for row in summary if row["mode"] == "iterative_vs_step13"]
    measures = [("k22/spatial_centered_cosine", "Partition K: centered cosine"),
                ("k15/spatial_centered_cosine", "Partition K L15H7: centered cosine"),
                ("semantic/content_heat_correlation", "Prompt heatmap correlation"),
                ("depth/spearman", "Relative depth: rank correlation"),
                ("qk_relation/changed_pixels/argmax_delta_px", "QK match disagreement (changed pixels)"),
                ("beam_paths/delta_to_step13_px", "Fixed-anchor beam path disagreement")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, measures):
        ax.plot([row["step"] for row in series], [row["metrics"][key] for row in series], marker="o")
        ax.set(title=title, xlabel="Denoising updates completed")
        ax.grid(alpha=0.25)
    fig.suptitle("Readout agreement with step 13; 33 observed + 16 privileged future frames")
    fig.savefig(output / "readout_convergence.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/vace_structured.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--direct-steps", type=int, nargs="*", default=[0, 4, 8, 13])
    args = parser.parse_args()
    config = PipelineConfig.load(args.config)
    config.teacher = dataclasses.replace(
        config.teacher, denoise_updates=config.teacher.denoise_reference_updates
    )
    if config.teacher.denoise_updates != 13 or any(not 0 <= step <= 13 for step in args.direct_steps):
        raise ValueError("This diagnostic compares the existing step-13 protocol")
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in Path(config.train.manifest).read_text().splitlines() if line]
    task_ids = sorted({row["task_index"] for row in rows})
    selected_tasks = [task_ids[int(index)] for index in np.linspace(0, len(task_ids) - 1, args.episodes)]
    items = []
    text_cache = torch.load(config.train.text_cache, map_location="cpu", weights_only=True)
    for task in selected_tasks:
        episode = next(row["episode"] for row in rows if row["task_index"] == task and row["length"] >= 80)
        for row in rows:
            if row["episode"] != episode:
                continue
            metadata = dict(row)
            start = max(0, row["length"] // 2 - 24)
            metadata.update(start=start, teacher_seed=stable_seed(config.train.seed, episode, row["camera"], start))
            indices = np.arange(row["episode_start_frame"] + start, row["episode_start_frame"] + start + 49)
            rgb = torch.from_numpy(decode_indices(row["video"], indices, row["fps"]))
            text = text_cache[row["task"]].float()
            items.append({"teacher_rgb": rgb, "text": text, "text_mask": text.abs().sum(-1) != 0,
                          "metadata": metadata})
    content = content_token_masks(config.teacher.text_root, [item["metadata"]["task"] for item in items])
    protocol = {"config": config.to_dict(), "samples": [item["metadata"] for item in items],
                "steps": list(range(14)), "direct_steps": args.direct_steps,
                "scope": "Future branch only. Native8 is observed boundary; native9..12 are future readouts.",
                "changed_pixels": "RGB difference >5/255 between adjacent native-time endpoints; not a GT object mask.",
                "reference": "Step13 teacher readout, not GT; agreement does not establish tracking accuracy.",
                "fixed_seeds": "Step13 prompt/K-selected anchors are shared by all decoders. Anchor locations excluded from path errors.",
                "noise_rng": "Identical per-window seeds and observed-branch RNG consumption to online teacher.",
                "direct_noise": "Same GT and original epsilon at the same sigma/timestep, with no iterative updates."}
    dump(args.output / "protocol.json", protocol)
    progress(stage="load_teacher", samples=len(items))
    teacher = VaceTeacher(config.teacher, "shared", device, text_cache[NEGATIVE_PROMPT])
    started = time.monotonic()
    records = []
    try:
        for begin in range(0, len(items), args.batch_size):
            batch = items[begin:begin + args.batch_size]
            snapshots, direct, schedule = trace(teacher, batch, 13, args.direct_steps)
            dump(args.output / f"schedule_batch{begin:02d}.json", schedule)
            for index, item in enumerate(batch):
                raw_rgb = item["teacher_rgb"]
                grid = config.teacher.image_size // 16
                reduced = F.adaptive_avg_pool2d(raw_rgb.float(), (grid, grid))
                energy = torch.stack([(reduced[t * 4] - reduced[(t - 1) * 4]).abs().mean(0).flatten()
                                      for t in range(9, 13)])
                reference = {name: value[index].float() for name, value in snapshots[-1].items()}
                content_mask = content[item["metadata"]["task"]]
                seeds = multi_anchor_paths(reference, FUTURE_ANCHORS, config.teacher, content_mask)["seeds"].numpy()
                reference_paths = fixed_paths(reference, seeds, config.teacher)
                source_size = raw_rgb.shape[-1]
                cell_px = source_size / grid
                comparisons = [("iterative_vs_step13", step, values)
                               for step, values in enumerate(snapshots)]
                comparisons += [("direct_vs_step13", step, values) for step, values in direct.items()]
                comparisons += [("direct_vs_same_step", step, values) for step, values in direct.items()]
                for mode, step, values in comparisons:
                    current = {name: value[index].float() for name, value in values.items()}
                    other = reference
                    other_paths = reference_paths
                    if mode == "direct_vs_same_step":
                        other = {name: value[index].float() for name, value in snapshots[step].items()}
                        other_paths = fixed_paths(other, seeds, config.teacher)
                    metrics = describe(current, other, energy, content_mask, seeds, other_paths,
                                       config.teacher, config.train.relation_temperature, cell_px)
                    records.append({"sample": item["metadata"]["id"], "mode": mode, "step": step,
                                    "sigma": schedule[step]["sigma"], "metrics": metrics})
                dump(args.output / "rows.json", records)
                progress(stage="sample_compared", sample=item["metadata"]["id"], rows=len(records))
            del snapshots, direct
    finally:
        teacher.close()
    summary = summarize(records)
    dump(args.output / "summary.json", summary)
    plot_summary(summary, args.output)
    dump(args.output / "completed.json", {"samples": len(items), "elapsed_seconds": time.monotonic() - started,
                                          "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30})
    progress(stage="completed", output=str(args.output), samples=len(items), elapsed_seconds=time.monotonic() - started)


if __name__ == "__main__":
    main()
