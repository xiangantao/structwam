#!/usr/bin/env python3
"""Capture untrained pre-RoPE K and full simulator instance labels for a scan."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "heft/diffusers/src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / "heft/latent_policy/.runtime/libero"))

import numpy as np
import torch
from torch.nn import functional as F

from render_libero_surface_tracks import (
    CAMERAS, OffScreenRenderEnv, benchmark, get_libero_path, render_buffers,
    restore, restore_fixed_body_poses, select_subjects,
)
from search_vace_tracking_heads import ControlTower, load_empty_text, load_vae, read_video


def log(**values):
    print(json.dumps(values), flush=True)


def render_instance_labels(directory, metadata, videos, frame_indices):
    suite = benchmark.get_benchmark_dict()[metadata["suite"]]()
    task = suite.get_task(int(metadata["task_id"]))
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    height, width = next(iter(videos.values())).shape[1:3]
    env = OffScreenRenderEnv(
        bddl_file_name=bddl, camera_names=list(videos), camera_heights=height,
        camera_widths=width, camera_depths=True, control_freq=int(metadata["fps"]),
        horizon=1000, ignore_done=True,
    )
    labels, errors = {camera: [] for camera in videos}, {camera: [] for camera in videos}
    try:
        env.reset()
        sim = env.env.sim
        if list(sim.model.body_names) != metadata["body_names"]:
            raise ValueError("Reconstructed simulation body names do not match")
        with np.load(directory / "sim_state.npz") as state:
            pose_error = restore_fixed_body_poses(sim, state)
            if pose_error > 1e-5:
                raise ValueError(f"Simulator pose alignment failed: {pose_error}")
            _, selected = select_subjects(sim, metadata["body_names"], True)
            body_labels = []
            for bid, name in enumerate(metadata["body_names"]):
                label = selected.get(bid)
                if label in {"gripper base", "left finger", "right finger"}:
                    label = "gripper"
                elif label == "wrist" or name.startswith("robot0_"):
                    label = "robot_arm"
                body_labels.append(label or "background")
            names = ["background", *sorted(set(body_labels) - {"background"})]
            body_ids = np.asarray([names.index(name) for name in body_labels], np.int16)
            geom_labels = body_ids[np.asarray(sim.model.geom_bodyid)]
            for frame in frame_indices:
                restore(sim, state, int(frame))
                for camera, video in videos.items():
                    rendered, _, geom_ids = render_buffers(sim, camera, height, width)
                    error = float(np.abs(rendered.astype(float) - video[frame].astype(float)).mean())
                    if error > 8:
                        raise ValueError(f"RGB replay does not align: {camera}, frame {frame}, MAE={error}")
                    pixel_labels = np.zeros_like(geom_ids, np.int16)
                    known = geom_ids >= 0
                    pixel_labels[known] = geom_labels[geom_ids[known]]
                    labels[camera].append(pixel_labels)
                    errors[camera].append(error)
    finally:
        env.close()
    return names, {camera: np.stack(values) for camera, values in labels.items()}, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=[39, 43, 83, 123])
    parser.add_argument("--wrist-episodes", type=int, nargs="*", default=[39, 83])
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--skip-frames", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=129)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    model_root = ROOT / "models/Wan2.1-VACE-1.3B"
    video_root = ROOT / "runs/vace_tracking_all_suites_20261004/videos"
    log(stage="load_text")
    text = load_empty_text(ROOT / "models/Wan2.1-T2V-1.3B-Diffusers", device)
    log(stage="load_tower")
    tower = ControlTower.load(model_root / "diffusion_pytorch_model.safetensors", device)
    vae = load_vae(model_root / "Wan2.1_VAE.pth", device)
    zero_latents, manifest = {}, []
    for episode in args.episodes:
        directory = video_root / f"episode_{episode:05d}"
        metadata = json.loads((directory / "metadata.json").read_text())
        cameras = ["agentview"] + (["robot0_eye_in_hand"] if episode in args.wrist_episodes else [])
        videos = {camera: read_video(directory / f"{camera}.mp4") for camera in cameras}
        n = min(len(videos[cameras[0]]) - args.skip_frames, args.max_frames)
        n = 1 + 4 * ((n - 1) // 4)
        times = np.arange(args.skip_frames, args.skip_frames + n, 4)
        sample_indices = np.unique(np.asarray([0, len(times) // 2, len(times) - 1]))
        frame_indices = times[sample_indices]
        names, masks, alignment = render_instance_labels(directory, metadata, videos, frame_indices)
        for camera, video in videos.items():
            name = f"episode_{episode:05d}_{camera}"
            log(stage="encode", case=name, frames=frame_indices.tolist(), suite=metadata["suite"])
            pixels = torch.from_numpy(video[args.skip_frames:args.skip_frames + n].copy()).to(device)
            pixels = pixels.permute(3, 0, 1, 2).float() / 127.5 - 1
            pixels = F.interpolate(pixels.permute(1, 0, 2, 3), size=(args.size, args.size),
                                   mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
            with torch.inference_mode():
                observed = vae.encode([pixels])[0]
                if n not in zero_latents:
                    zero_latents[n] = vae.encode([torch.zeros_like(pixels)])[0]
                reactive = zero_latents[n]
                control = torch.cat((observed, reactive, torch.zeros(
                    (64, *observed.shape[1:]), device=device)), dim=0).unsqueeze(0).bfloat16()
                generator = torch.Generator(device=device).manual_seed(42 + episode * 100 + CAMERAS.index(camera))
                noise = torch.randn((1, *observed.shape), generator=generator, device=device).bfloat16()
                keys, handles = {}, []
                latent_frames, grid = len(times), args.size // 16
                def make_hook(layer):
                    def hook(module, inputs, output):
                        values = output[0].reshape(latent_frames, grid * grid, 12, 128)
                        keys[f"layer_{layer}_k"] = values[sample_indices].permute(2, 0, 1, 3).float().cpu().numpy().astype(np.float16)
                    return hook
                try:
                    for layer, block in enumerate(tower.vace_blocks):
                        handles.append(block.attn1.norm_k.register_forward_hook(make_hook(layer)))
                    tower(noise, control, text, torch.tensor([1000], device=device), lambda *_: None)
                finally:
                    for handle in handles:
                        handle.remove()
            np.savez(args.output / f"{name}.npz", **keys, frame_indices=frame_indices,
                     rgb=video[frame_indices], instance_labels=masks[camera], label_names=np.asarray(names))
            row = {"name": name, "episode": episode, "camera": camera, "suite": metadata["suite"],
                   "task": metadata["task"], "frame_indices": frame_indices.tolist(),
                   "rgb_replay_mae": alignment[camera], "layers": len(keys),
                   "rgb_window_frames": n, "grid": grid}
            manifest.append(row)
            (args.output / "manifest.json").write_text(json.dumps({"cases": manifest}, indent=2))
            log(stage="saved", **row)
            del pixels, observed, reactive, control, noise, keys
            gc.collect()
            torch.cuda.empty_cache()
    log(stage="done", cases=len(manifest), output=str(args.output))


if __name__ == "__main__":
    main()
