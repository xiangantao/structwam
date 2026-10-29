#!/usr/bin/env python3
"""Find native main-tower attention that recruits a grasped object.

Simulator masks choose gripper query rows and measure regional responses. They
never change the keys, logits, probabilities, or displayed attention values.
The teacher input is the established 33 observed + 16 GT-noised frame protocol.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from scan_vace_interaction_heads import (
    NEGATIVE_PROMPT,
    OBSERVED_FRAMES,
    ROOT,
    TOTAL_FRAMES,
    UniPCMultistepScheduler,
    dump,
    encode_prompts,
    load_transformer,
    load_vae,
    log,
    prepare_inputs,
    read_video,
    remove_hooks,
    stable_seed,
)
from search_vace_tracking_heads import write_video


TIMES = np.arange(0, TOTAL_FRAMES, 4)
REGIONS = ("background", "gripper", "target", "other_objects")
DEFAULT_SAMPLES = (
    "sample_00008", "sample_00006",  # bowl, spatial task 1
    "sample_00051", "sample_00049",  # cream cheese, object task 1
    "sample_00060", "sample_00066",  # sauce, object task 3
    "sample_00121", "sample_00123",  # cream cheese, goal task 6
)


def read_record(data: Path, sample: str) -> dict:
    record = json.loads((data / sample / "metadata.json").read_text())
    if record["event_type"] != "grasp" or record["num_frames"] != TOTAL_FRAMES:
        raise ValueError(f"Expected a 49-frame grasp window: {sample}")
    return record


def native_state_curves(state: dict) -> dict:
    object_id, hand_id = int(state["target_body_index"]), int(state["hand_body_index"])
    obj = state["body_xpos"][:, object_id]
    hand = state["body_xpos"][:, hand_id]
    rotation = state["body_xmat"][:, hand_id].reshape(-1, 3, 3)
    relative = np.einsum("ti,tij->tj", obj - hand, rotation)
    baseline = float(np.median(obj[24:33, 2]))
    return {
        "object_lift_m": (obj[:, 2] - baseline).tolist(),
        "hand_object_distance_m": np.linalg.norm(obj - hand, axis=-1).tolist(),
        "object_in_hand_frame_m": relative.tolist(),
        "max_lift_in_saved_outcome_m": float((obj[33:, 2] - baseline).max()),
        "tail_hand_object_distance_m": float(np.median(np.linalg.norm(obj[-4:] - hand[-4:], axis=-1))),
        "label_note": "Original v4 labels are heuristics; these curves use only the saved 49 frames.",
    }


def label_panel(image: np.ndarray, text: str) -> np.ndarray:
    result = cv2.copyMakeBorder(image, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    cv2.putText(result, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1, cv2.LINE_AA)
    return result


def annotate(args, records: list[dict]) -> None:
    from collect_libero_long_videos import make_env
    from render_libero_surface_tracks import render_buffers, restore, restore_fixed_body_poses, select_subjects

    grid = args.size // 16
    for record in records:
        directory = args.output / "samples" / record["sample_id"]
        directory.mkdir(parents=True, exist_ok=True)
        if args.resume and (directory / "regions.npz").exists():
            continue
        with np.load(args.data / record["sample_id"] / "state.npz") as archive:
            state = {name: archive[name] for name in archive.files}
        frames = read_video(args.data / record["sample_id"] / "wrist.mp4")
        height, width = frames.shape[1:3]
        if height != width or height % grid:
            raise ValueError("The recorded square image must divide into the model attention grid")
        _, _, env, _ = make_env(record["suite"], record["task_id"], height, width, 20, TOTAL_FRAMES)
        try:
            env.reset()
            sim = env.env.sim
            pose_error = restore_fixed_body_poses(sim, state)
            if pose_error > 0.001:
                raise RuntimeError(f"Simulator restoration differs by {pose_error:.6f} m")
            names = list(sim.model.body_names)
            _, labels = select_subjects(sim, names, include_objects=True)
            gripper_ids = [bid for bid, label in labels.items() if label in {"gripper base", "left finger", "right finger"}]
            target_id = names.index(record["target_body"])
            body_region = np.zeros(sim.model.nbody, np.int32)
            for body in range(1, sim.model.nbody):
                ancestor = body
                while ancestor and ancestor != target_id and not names[ancestor].endswith("_main"):
                    ancestor = int(sim.model.body_parentid[ancestor])
                if ancestor == target_id:
                    body_region[body] = 2
                elif ancestor and names[ancestor].endswith("_main"):
                    body_region[body] = 3
            body_region[gripper_ids] = 1
            masks, occupancy, tiles, image_errors = [], [], [], []
            for native_time in TIMES:
                restore(sim, state, int(native_time))
                rgb, _, geometry = render_buffers(sim, "robot0_eye_in_hand", height, width)
                image_errors.append(float(np.abs(rgb.astype(float) - frames[native_time]).mean()))
                valid = geometry >= 0
                region = np.zeros((height, width), np.uint8)
                region[valid] = body_region[np.asarray(sim.model.geom_bodyid)[geometry[valid]]]
                masks.append(region)
                scale = height // grid
                weights = np.stack([
                    (region == index).reshape(grid, scale, grid, scale).mean((1, 3))
                    for index in range(len(REGIONS))
                ])
                occupancy.append(weights)
                if native_time in (0, 24, 32, 40, 48):
                    overlay = frames[native_time].copy()
                    for index, color in ((1, (45, 210, 255)), (2, (255, 170, 40)), (3, (150, 100, 230))):
                        overlay[region == index] = np.rint(0.6 * overlay[region == index] + 0.4 * np.array(color))
                    tiles.append(label_panel(overlay, f"{record['sample_id']} f{native_time} label={record['label']}"))
            occupancy = np.asarray(occupancy, np.float32)
            # Only substantially gripper-covered cells supply query rows.
            query_cells = np.zeros((len(TIMES), args.max_queries), np.int64)
            query_weights = np.zeros_like(query_cells, dtype=np.float32)
            for t, weight in enumerate(occupancy[:, 1].reshape(len(TIMES), -1)):
                ids = np.flatnonzero(weight >= args.min_gripper_coverage)
                if len(ids) > args.max_queries:
                    ids = ids[np.linspace(0, len(ids) - 1, args.max_queries).round().astype(int)]
                query_cells[t, :len(ids)] = ids
                query_weights[t, :len(ids)] = weight[ids]
            if np.count_nonzero(query_weights.sum(1)) < 10:
                raise RuntimeError(f"Too few visible gripper-query frames: {record['sample_id']}")
            np.savez_compressed(directory / "regions.npz", native_times=TIMES, masks=np.stack(masks),
                                occupancy=occupancy, query_cells=query_cells, query_weights=query_weights)
            sheet = np.concatenate(tiles, axis=1)
            cv2.imwrite(str(directory / "region_check.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
            dump(directory / "metadata.json", record | {
                "curves": native_state_curves(state), "restored_body_error_m": pose_error,
                "render_rgb_mean_absolute_error": image_errors,
                "query_body_names": [names[bid] for bid in gripper_ids],
                "query_count_per_latent_time": (query_weights > 0).sum(1).tolist(),
                "region_names": list(REGIONS),
            })
            log(stage="regions_ready", sample=record["sample_id"], query_counts=(query_weights > 0).sum(1).tolist(),
                image_mae=float(np.mean(image_errors)))
        finally:
            env.close()


class GripperCapture:
    def __init__(self, regions: Path, device: torch.device):
        with np.load(regions) as archive:
            self.cells = torch.as_tensor(archive["query_cells"], device=device)
            self.weights = torch.as_tensor(archive["query_weights"], device=device)
            self.grid = int(archive["occupancy"].shape[-1])
        self.enabled = False
        self.maps: dict[int, np.ndarray] = {}
        self.temporal_mass: dict[int, np.ndarray] = {}

    def clear(self, enabled: bool) -> None:
        self.enabled = enabled
        self.maps.clear()
        self.temporal_mass.clear()

    def __call__(self, layer: int, query: torch.Tensor, key: torch.Tensor, **_) -> None:
        if not self.enabled:
            return
        temporal, queries = self.cells.shape
        spatial = self.grid ** 2
        heads, tokens, dim = query.shape[1:]
        if query.shape != key.shape or tokens != temporal * spatial or query.shape[0] != 1:
            raise RuntimeError("Unexpected main Q/K shape")
        indices = (torch.arange(temporal, device=query.device)[:, None] * spatial + self.cells).flatten()
        q = query[0, :, indices].float()
        logits = torch.matmul(q, key[0].float().transpose(-1, -2)) / math.sqrt(dim)
        # Normalize over the complete native spatiotemporal key sequence.
        attention = logits.softmax(-1).reshape(heads, temporal, queries, temporal, spatial)
        weights = self.weights / self.weights.sum(1, keepdim=True).clamp_min(1e-12)
        pooled = torch.einsum("htqks,tq->htks", attention, weights)
        time_index = torch.arange(temporal, device=query.device)
        same_frame = pooled[:, time_index, time_index]
        mass = pooled.sum(-1)
        invalid = self.weights.sum(1) == 0
        same_frame[:, invalid] = float("nan")
        mass[:, invalid] = float("nan")
        if not (torch.nansum(same_frame, dim=-1) <= 1.00001).all():
            raise RuntimeError("The same-frame slice exceeds native attention mass")
        self.maps[layer] = same_frame.reshape(heads, temporal, self.grid, self.grid).cpu().numpy().astype(np.float32)
        self.temporal_mass[layer] = mass.cpu().numpy().astype(np.float32)


@torch.inference_mode()
def extract(args, records: list[dict]) -> None:
    device = torch.device(args.device)
    torch.set_num_threads(4)
    prompts = encode_prompts(args.text_model, [row["task"] for row in records], device)
    model = load_transformer(args.model, device)
    vae = load_vae(args.model / "Wan2.1_VAE.pth", device)
    scheduler = UniPCMultistepScheduler.from_pretrained(args.text_model / "scheduler", local_files_only=True, flow_shift=3)
    states = sorted(set(args.states))
    started = time.monotonic()
    for index, record in enumerate(records):
        directory = args.output / "samples" / record["sample_id"]
        if args.resume and (directory / "attention.npz").exists():
            continue
        capture = GripperCapture(directory / "regions.npz", device)
        for layer, block in enumerate(model.blocks):
            block.attn1._heft_capture = lambda layer=layer, **kw: capture(layer, kw["query"], kw["key"])
        try:
            truth = read_video(args.data / record["sample_id"] / "wrist.mp4")
            control, clean = prepare_inputs(vae, truth, args.size, OBSERVED_FRAMES, device)
            scheduler.set_timesteps(40, device=device)
            begin = int((scheduler.sigmas[:-1] - 0.8).abs().argmin())
            scheduler.set_begin_index(begin)
            timesteps = scheduler.timesteps[begin:]
            if states[0] < 0 or states[-1] >= len(timesteps):
                raise ValueError("Captured denoising states are outside the schedule")
            generator = torch.Generator(device=device).manual_seed(stable_seed(args.seed, record["sample_id"]))
            noise = torch.randn(clean.shape, device=device, generator=generator)
            latents = scheduler.add_noise(clean, noise, timesteps[:1])
            positive = prompts[record["task"]].to(device)
            negative = prompts[NEGATIVE_PROMPT].to(device)
            scales = torch.ones(len(model.vace_blocks), device=device, dtype=torch.bfloat16)
            maps, temporal_masses = [], []
            for state, timestep in enumerate(timesteps):
                capture.clear(state in states)
                inputs = {
                    "hidden_states": latents.bfloat16(), "timestep": timestep.reshape(1),
                    "control_hidden_states": control, "control_hidden_states_scale": scales, "return_dict": False,
                }
                conditional = model(**inputs, encoder_hidden_states=positive)[0]
                if state in states:
                    if sorted(capture.maps) != list(range(len(model.blocks))):
                        raise RuntimeError("Some main self-attention hooks were not captured")
                    maps.append(np.stack([capture.maps[layer] for layer in range(len(model.blocks))]))
                    temporal_masses.append(np.stack([capture.temporal_mass[layer] for layer in range(len(model.blocks))]))
                    log(stage="attention_captured", sample=record["sample_id"], state=state,
                        elapsed_s=round(time.monotonic() - started, 1))
                if state >= states[-1]:
                    break
                capture.clear(False)
                unconditional = model(**inputs, encoder_hidden_states=negative)[0]
                flow = unconditional.float() + 5 * (conditional.float() - unconditional.float())
                latents = scheduler.step(flow, timestep, latents, return_dict=False)[0]
            np.savez_compressed(directory / "attention.npz", states=np.asarray(states), native_times=TIMES,
                                same_frame=np.stack(maps), temporal_mass=np.stack(temporal_masses))
            log(stage="sample_complete", sample=record["sample_id"], completed=index + 1, total=len(records),
                elapsed_s=round(time.monotonic() - started, 1), actual_sigma=float(scheduler.sigmas[begin]))
            del control, clean, noise, latents, conditional, positive, negative
        finally:
            remove_hooks(model)
            capture.clear(False)
            gc.collect()
            torch.cuda.empty_cache()
    del model, vae, prompts
    gc.collect()
    torch.cuda.empty_cache()


def response_curves(maps: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
    # Area-weighted mean attention per pixel avoids rewarding larger regions.
    masks = occupancy.reshape(len(TIMES), len(REGIONS), -1).copy()
    # Do not count gripper response inside a mixed token as object recruitment.
    masks[:, 2] *= masks[:, 1] <= 0.05
    weights = masks / masks.sum(-1, keepdims=True).clip(1e-12)
    values = np.einsum("lhts,trs->lhtr", maps.reshape(*maps.shape[:2], len(TIMES), -1), weights)
    values[..., masks.sum(-1) == 0] = np.nan
    return values


def heat_overlay(frame: np.ndarray, heat: np.ndarray, ceiling: float, log_floor: float | None = None) -> np.ndarray:
    if log_floor is None:
        normalized = heat / max(ceiling, 1e-12)
    else:
        normalized = np.log1p(heat / log_floor) / np.log1p(ceiling / log_floor)
    normalized = np.nan_to_num(normalized).clip(0, 1)
    # Keep the actual token grid visible; interpolation must not invent detail.
    normalized = cv2.resize(normalized, frame.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    color = cv2.cvtColor(cv2.applyColorMap(np.rint(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    alpha = (0.62 * normalized)[..., None]
    return np.rint(frame * (1 - alpha) + color * alpha).astype(np.uint8)


def finite_mean(values) -> float:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def report(args, records: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples, pairs = {}, {}
    for record in records:
        directory = args.output / "samples" / record["sample_id"]
        with np.load(directory / "attention.npz") as archive:
            maps = archive["same_frame"]
            states = archive["states"]
        with np.load(directory / "regions.npz") as archive:
            occupancy = archive["occupancy"]
        curves = np.stack([response_curves(m, occupancy) for m in maps])
        samples[record["sample_id"]] = (maps, curves)
        key = (record["suite"], record["task_id"])
        pairs.setdefault(key, {})[record["label"]] = record["sample_id"]
    if any(set(pair) != {0, 1} for pair in pairs.values()):
        raise ValueError("Every diagnostic task must have one success and one failure")
    early, late = TIMES <= 24, TIMES >= 40
    results = []
    for si, state in enumerate(states):
        for layer in range(maps.shape[1]):
            for head in range(maps.shape[2]):
                details = []
                for task, pair in pairs.items():
                    summaries = {}
                    for label, sample in pair.items():
                        c = samples[sample][1][si, layer, head]
                        ratio = c[:, 2] / c[:, 0].clip(1e-12)
                        before, after = finite_mean(ratio[early]), finite_mean(ratio[late])
                        summaries[str(label)] = {
                            "sample": sample, "early_target_to_background": before,
                            "late_target_to_background": after, "target_gain": after / max(before, 1e-12),
                            "early_gripper_to_target": finite_mean(c[early, 1] / c[early, 2].clip(1e-12)),
                            "late_gripper_to_background": finite_mean(c[late, 1] / c[late, 0].clip(1e-12)),
                        }
                    positive, negative = summaries["1"], summaries["0"]
                    contrast = math.log(max(positive["target_gain"], 1e-12) / max(negative["target_gain"], 1e-12))
                    pattern = (positive["early_gripper_to_target"] >= 2
                               and positive["late_gripper_to_background"] >= 2
                               and positive["late_target_to_background"] >= 2
                               and positive["target_gain"] >= 2
                               and negative["target_gain"] <= 1.25
                               and positive["late_target_to_background"] >= 1.5 * negative["late_target_to_background"])
                    details.append({"suite": task[0], "task_id": task[1], "curves": summaries,
                                    "gain_contrast_log": contrast, "matches_fixed_pattern": bool(pattern)})
                if not all(math.isfinite(d["gain_contrast_log"]) for d in details):
                    continue
                results.append({"state_after_steps": int(state), "layer": layer, "head": head,
                                "matching_pairs": sum(d["matches_fixed_pattern"] for d in details),
                                "mean_gain_contrast_log": float(np.mean([d["gain_contrast_log"] for d in details])),
                                "worst_pair_gain_contrast_log": float(min(d["gain_contrast_log"] for d in details)),
                                "pairs": details})
    results.sort(key=lambda r: (-r["matching_pairs"], -r["worst_pair_gain_contrast_log"], -r["mean_gain_contrast_log"]))
    dump(args.output / "ranking.json", results)
    dump(args.output / "summary.json", {
        "samples": len(records), "matched_tasks": len(pairs), "heads_and_states_scanned": len(results),
        "heads_matching_all_pairs": sum(r["matching_pairs"] == len(pairs) for r in results),
        "max_matching_pairs": max(r["matching_pairs"] for r in results), "top": results[:args.top],
        "interpretation": "Screening counts are exploratory thresholds, not evidence of a clear interaction detector; object response remains much weaker than gripper response.",
        "selection": "Exploratory ranking on these diagnostic pairs; no trained readout.",
        "pattern": "Gripper-only early; selective target recruitment and continued gripper response late; weak recruitment in matched failure.",
    })
    for candidate in results[:args.top]:
        si = int(np.flatnonzero(states == candidate["state_after_steps"])[0])
        layer, head = candidate["layer"], candidate["head"]
        tag = f"step{candidate['state_after_steps']}_L{layer}H{head}"
        preview = args.output / "previews" / tag
        preview.mkdir(parents=True, exist_ok=True)
        all_heat = np.stack([samples[r["sample_id"]][0][si, layer, head] for r in records])
        ceiling = float(np.nanpercentile(all_heat, 99.5))
        log_floor = 1 / (len(TIMES) * (args.size // 16) ** 2 * 100)
        all_curves = np.stack([samples[r["sample_id"]][1][si, layer, head] for r in records])
        ratio_ceiling = float(np.nanmax(all_curves[..., 1:] / all_curves[..., :1].clip(1e-12)))
        figure, axes = plt.subplots(len(pairs), 2, figsize=(12, 3 * len(pairs)), squeeze=False)
        for pi, (task, pair) in enumerate(pairs.items()):
            for column, label in enumerate((1, 0)):
                sample = pair[label]
                curves = samples[sample][1][si, layer, head]
                ax = axes[pi, column]
                for region, color in ((1, "#168bb4"), (2, "#e07018"), (3, "#8866aa")):
                    ax.plot(TIMES, curves[:, region] / curves[:, 0].clip(1e-12), marker="o", label=REGIONS[region], color=color)
                ax.axvline(32, color="black", linestyle="--", linewidth=1)
                ax.axhline(1, color="grey", linewidth=0.7)
                ax.set_yscale("symlog", linthresh=1)
                ax.set_ylim(0, ratio_ceiling * 1.1)
                ax.set(title=f"{sample} {'success label' if label else 'failure label'} / {task[0]} {task[1]}",
                       xlabel="Native frame (20 Hz)", ylabel="Mean attention / background")
                ax.legend(fontsize=8)
                record = next(r for r in records if r["sample_id"] == sample)
                frames = read_video(args.data / sample / "wrist.mp4")
                heat = samples[sample][0][si, layer, head]
                rendered, tiles, log_tiles = [], [], []
                for t, frame in enumerate(frames):
                    # Render the nearest available latent time, with no trajectory smoothing.
                    ti = int(np.abs(TIMES - t).argmin())
                    rgb = label_panel(frame, f"{sample} f{t} label={label}")
                    short = f"s{candidate['state_after_steps']} L{layer}H{head} f{t}"
                    visual = label_panel(heat_overlay(frame, heat[ti], ceiling), f"{short} linear")
                    log_visual = label_panel(heat_overlay(frame, heat[ti], ceiling, log_floor), f"{short} log")
                    rendered.append(np.concatenate([rgb, visual, log_visual], axis=1))
                    if t in (0, 24, 32, 40, 48):
                        tiles.append(visual)
                        log_tiles.append(log_visual)
                write_video(preview / f"{sample}.mp4", np.stack(rendered), record["output_fps"])
                cv2.imwrite(str(preview / f"{sample}.png"), cv2.cvtColor(np.concatenate(tiles, axis=1), cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(preview / f"{sample}_log.png"), cv2.cvtColor(np.concatenate(log_tiles, axis=1), cv2.COLOR_RGB2BGR))
        figure.suptitle(tag)
        figure.tight_layout()
        figure.savefig(preview / "response_curves.png", dpi=160)
        plt.close(figure)
        dump(preview / "color_scale.json", {"native_probability_min": 0, "native_probability_max": ceiling,
                                              "log_display_floor": log_floor,
                                              "log_display_formula": "log1p(p / floor) / log1p(ceiling / floor)",
                                              "shared_across_all_samples_and_times": True})
    log(stage="report_complete", top=results[:1], output=str(args.output))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "runs/libero_interaction_probe_33plus16_20261008_all_native20_v4")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "models/Wan2.1-VACE-1.3B")
    parser.add_argument("--text-model", type=Path, default=ROOT / "models/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--samples", nargs="+", default=list(DEFAULT_SAMPLES))
    parser.add_argument("--phase", choices=("annotate", "extract", "report", "all"), default="all")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--states", type=int, nargs="+", default=[0, 13, 22])
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--max-queries", type=int, default=16)
    parser.add_argument("--min-gripper-coverage", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.size % 16 or not args.states or args.max_queries < 1:
        parser.error("size must divide by 16, states must be present, and max-queries must be positive")
    args.data = args.data.resolve(strict=True)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    records = [read_record(args.data, sample) for sample in args.samples]
    dump(args.output / "protocol.json", {
        "input": "33 observed wrist frames, no warmup; full 49-frame GT main latent + sigma~0.8; full task prompt",
        "noise_seed": args.seed, "CFG": 5, "scheduler": "UniPC 40 steps, flow shift 3",
        "states_after_completed_updates": sorted(set(args.states)), "main_attn1_only": True,
        "queries": "Visible gripper-covered tokens, updated from simulator masks at every nominal latent frame",
        "max_queries": args.max_queries, "min_gripper_coverage": args.min_gripper_coverage,
        "attention": "softmax(post-RoPE Q K^T / sqrt(d)) over all native time-space keys; show same-time slice",
        "native_times": TIMES.tolist(), "time_alignment": "Nominal 4-frame VAE spacing; no claim of per-RGB-frame resolution",
        "metrics": "Mean attention per region pixel, target/background gain, matched success-failure gain difference; target excludes cells with >5% gripper coverage",
        "masks": "Diagnostic query selection and regional scoring only; no mask-based expansion of heat",
        "visualization": "One fixed color range per candidate, shared across all frames and success/failure samples",
        "selection": "Four matched task pairs chosen from saved replay state before attention scanning; exploratory",
        "records": records,
    })
    if args.phase in ("annotate", "all"):
        annotate(args, records)
    if args.phase in ("extract", "all"):
        extract(args, records)
    if args.phase in ("report", "all"):
        report(args, records)


if __name__ == "__main__":
    main()
