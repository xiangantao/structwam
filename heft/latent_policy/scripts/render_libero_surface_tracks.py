#!/usr/bin/env python3
"""Render rigid surface-point tracks with per-camera occlusion masks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import av
import cv2
import mujoco
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from lerobot.utils.io_utils import write_video
from robosuite.utils.camera_utils import get_camera_transform_matrix, get_real_depth_map

CAMERAS = ("robot0_eye_in_hand", "agentview")
BODY_COLORS = {
    "robot0_link6": (255, 190, 50),
    "robot0_link7": (255, 190, 50),
    "robot0_right_hand": (230, 100, 230),
    "gripper0_right_gripper": (230, 100, 230),
    "gripper0_leftfinger": (70, 230, 120),
    "gripper0_finger_joint1_tip": (70, 230, 120),
    "gripper0_rightfinger": (80, 160, 255),
    "gripper0_finger_joint2_tip": (80, 160, 255),
}
OBJECT_PALETTE = (
    (255, 110, 60), (60, 210, 230), (250, 215, 70), (190, 150, 255),
    (180, 245, 95), (95, 170, 255), (255, 150, 190), (170, 210, 180),
)


def select_subjects(sim, names, include_objects):
    colors = {i: BODY_COLORS[name] for i, name in enumerate(names) if name in BODY_COLORS}
    labels = {}
    for bid in colors:
        name = names[bid]
        if "finger_joint1" in name or "leftfinger" in name:
            labels[bid] = "left finger"
        elif "finger_joint2" in name or "rightfinger" in name:
            labels[bid] = "right finger"
        elif "gripper" in name or name == "robot0_right_hand":
            labels[bid] = "gripper base"
        else:
            labels[bid] = "wrist"
    if include_objects:
        roots = {i: name.removesuffix("_main") for i, name in enumerate(names) if name.endswith("_main")}
        root_colors = {bid: OBJECT_PALETTE[i % len(OBJECT_PALETTE)] for i, bid in enumerate(roots)}
        for bid in range(1, len(names)):
            ancestor = bid
            while ancestor != 0 and ancestor not in roots:
                ancestor = int(sim.model.body_parentid[ancestor])
            if ancestor in roots:
                colors[bid] = root_colors[ancestor]
                labels[bid] = roots[ancestor]
    return colors, labels


def read_frames(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def restore(sim, state, t: int) -> None:
    sim.data.qpos[:] = state["qpos"][t]
    sim.data.qvel[:] = state["qvel"][t]
    sim.forward()


def restore_fixed_body_poses(sim, state) -> float:
    positions = state["body_xpos"][0].astype(np.float64)
    rotations = state["body_xmat"][0].astype(np.float64).reshape(-1, 3, 3)
    # qpos does not store randomized poses of bodies attached without a joint.
    for bid in range(1, sim.model.nbody):
        if sim.model.body_jntnum[bid] != 0:
            continue
        parent = sim.model.body_parentid[bid]
        local_pos = (positions[bid] - positions[parent]) @ rotations[parent]
        local_rot = rotations[parent].T @ rotations[bid]
        if np.linalg.norm(local_pos - sim.model.body_pos[bid]) > 1e-5:
            sim.model.body_pos[bid] = local_pos
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, local_rot.ravel())
        current = sim.model.body_quat[bid]
        if min(np.linalg.norm(quat - current), np.linalg.norm(quat + current)) > 1e-5:
            sim.model.body_quat[bid] = quat
    restore(sim, state, 0)
    return float(np.linalg.norm(sim.data.body_xpos - positions, axis=1).max())


def render_buffers(sim, camera: str, height: int, width: int):
    rgb, depth = sim.render(camera_name=camera, height=height, width=width, depth=True)
    depth = get_real_depth_map(sim, depth[::-1]).astype(np.float32)
    context = sim._render_context_offscreen
    context.render(width, height, camera_id=sim.model.camera_name2id(camera), segmentation=True)
    # The installed robosuite segmentation decoder overflows uint8 with NumPy 2.
    encoded = context.read_pixels(width, height)[::-1].astype(np.int32)
    codes = encoded[:, :, 0] + (encoded[:, :, 1] << 8) + (encoded[:, :, 2] << 16)
    lookup = np.full(context.scn.ngeom + 1, -1, dtype=np.int32)
    for geom in context.scn.geoms[: context.scn.ngeom]:
        if geom.segid >= 0 and geom.objtype == mujoco.mjtObj.mjOBJ_GEOM:
            lookup[geom.segid + 1] = geom.objid
    geom_ids = lookup[np.clip(codes, 0, len(lookup) - 1)]
    return rgb[::-1].copy(), depth, geom_ids


def surface_samples(sim, depth, geom_ids, camera, selected, stride, voxel):
    height, width = depth.shape
    yy, xx = np.mgrid[1:height:stride, 1:width:stride]
    yy, xx = yy.ravel(), xx.ravel()
    gids = geom_ids[yy, xx]
    valid = gids >= 0
    bids = np.full(len(gids), -1, dtype=np.int32)
    bids[valid] = sim.model.geom_bodyid[gids[valid]]
    valid &= np.isin(bids, list(selected))
    xx, yy, bids, gids = xx[valid], yy[valid], bids[valid], gids[valid]
    z = depth[yy, xx]
    # MuJoCo raster samples are at pixel centers; its intrinsic matrix uses edges.
    homogeneous = np.c_[(xx + 0.5) * z, (yy + 0.5) * z, z, np.ones(len(z))]
    matrix = get_camera_transform_matrix(sim, camera, height, width)
    world = (np.linalg.inv(matrix) @ homogeneous.T).T[:, :3]
    rotations = sim.data.body_xmat[bids].reshape(-1, 3, 3)
    local = np.einsum("nj,njk->nk", world - sim.data.body_xpos[bids], rotations)
    cells = np.rint(local / voxel).astype(np.int64)
    keys = np.c_[bids, cells]
    return local, bids, gids, keys


def project_visible(sim, camera, world, point_geom_ids, depth, geom_ids, atol, rtol):
    height, width = depth.shape
    matrix = get_camera_transform_matrix(sim, camera, height, width)
    hp = (matrix @ np.c_[world, np.ones(len(world))].T).T
    xy = np.full((len(world), 2), np.nan, dtype=np.float32)
    positive = hp[:, 2] > 0
    xy[positive] = hp[positive, :2] / hp[positive, 2:3] - 0.5
    uv = np.zeros((len(world), 2), dtype=np.int32)
    uv[positive] = np.floor(xy[positive] + 0.5).astype(np.int32)
    near = sim.model.vis.map.znear * sim.model.stat.extent
    far = sim.model.vis.map.zfar * sim.model.stat.extent
    in_frame = positive & (hp[:, 2] >= near) & (hp[:, 2] <= far)
    in_frame &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    visible = np.zeros(len(world), dtype=bool)
    indices = np.flatnonzero(in_frame)
    pixels = uv[indices]
    surface_z = depth[pixels[:, 1], pixels[:, 0]]
    same_geom = geom_ids[pixels[:, 1], pixels[:, 0]] == point_geom_ids[indices]
    errors = np.abs(hp[indices, 2] - surface_z)
    visible[indices] = same_geom & (errors <= atol + rtol * surface_z)
    return xy, visible, in_frame, hp[:, 2].astype(np.float32)


def overlay(frames, xy, visible, body_ids, colors, tail=0, trail_indices=None):
    output = []
    if trail_indices is None:
        trail_indices = np.arange(xy.shape[1])
    trail_colors = np.asarray([colors[int(body_ids[i])] for i in trail_indices])
    for t, frame in enumerate(frames):
        image = frame.copy()
        if tail > 1 and t > 0:
            start = max(0, t - tail + 1)
            # Only the uninterrupted visible segment ending at this frame survives.
            mask = visible[start : t + 1, trail_indices]
            continuous = np.logical_and.accumulate(mask[::-1], axis=0)[::-1]
            for j in range(start + 1, t + 1):
                previous = xy[j - 1, trail_indices]
                current = xy[j, trail_indices]
                draw = continuous[j - 1 - start] & continuous[j - start]
                draw &= np.linalg.norm(current - previous, axis=1) >= 0.5
                if not np.any(draw):
                    continue
                layer = image.copy()
                for a, b, color in zip(previous[draw], current[draw], trail_colors[draw], strict=True):
                    cv2.line(layer, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)),
                             tuple(int(v) for v in color), 1, cv2.LINE_AA)
                alpha = 0.10 + 0.55 * (j - start) / (t - start)
                image = cv2.addWeighted(layer, alpha, image, 1 - alpha, 0)
        dots = image.copy()
        for bid in np.unique(body_ids):
            selected = (body_ids == bid) & visible[t]
            pixels = np.rint(xy[t, selected]).astype(int)
            for x, y in pixels:
                cv2.circle(dots, (int(x), int(y)), 1, colors[int(bid)], -1, cv2.LINE_AA)
        output.append(cv2.addWeighted(dots, 0.65, image, 0.35, 0))
    return output


def contact_sheet(frames, path):
    indices = np.linspace(0, len(frames) - 1, 6, dtype=int)
    tiles = []
    for t in indices:
        tile = cv2.copyMakeBorder(frames[t], 24, 0, 0, 0, cv2.BORDER_CONSTANT, value=(35, 35, 35))
        cv2.putText(tile, f"frame {t}", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
        tiles.append(tile)
    sheet = np.concatenate([np.concatenate(tiles[:3], axis=1), np.concatenate(tiles[3:], axis=1)])
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def paired_views(rendered):
    pairs = []
    for agent, wrist in zip(rendered["agentview"], rendered["robot0_eye_in_hand"], strict=True):
        panel = np.concatenate([agent, wrist], axis=1)
        panel = cv2.copyMakeBorder(panel, 24, 0, 0, 0, cv2.BORDER_CONSTANT, value=(35, 35, 35))
        for x, name in ((8, "agentview"), (agent.shape[1] + 8, "wrist camera")):
            cv2.putText(panel, name, (x, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
        pairs.append(panel)
    return pairs


def save_legend(colors, labels, path):
    entries = {labels[bid]: color for bid, color in colors.items()}
    image = np.full((28 * len(entries) + 12, 440, 3), 35, dtype=np.uint8)
    for i, (name, color) in enumerate(entries.items()):
        y = 25 + i * 28
        name = name.replace("akita_black_bowl", "black bowl").replace("glazed_rim_porcelain_ramekin", "ramekin")
        name = name.replace("wooden_cabinet", "cabinet").replace("flat_stove", "stove").replace("_", " ")
        cv2.rectangle(image, (12, y - 12), (26, y + 2), color, -1)
        scale = 0.5
        text_width = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        if text_width > 380:
            scale *= 380 / text_width
        cv2.putText(image, name, (38, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=3)
    parser.add_argument("--episodes", type=int, nargs="+", help="Select episode IDs from the manifest")
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--seed-stride", type=int, default=16)
    parser.add_argument("--voxel-mm", type=float, default=2.0)
    parser.add_argument("--depth-atol-mm", type=float, default=1.0)
    parser.add_argument("--depth-rtol", type=float, default=0.001)
    parser.add_argument("--include-objects", action="store_true", help="Track scene objects and their articulated child bodies too")
    parser.add_argument("--annotations-only", action="store_true", help="Save point labels without rendering preview videos")
    parser.add_argument("--frame-stride", type=int, default=1, help="Render visibility only at these RGB frame intervals")
    parser.add_argument("--resume", action="store_true", help="Reuse completed annotations")
    args = parser.parse_args()
    if min(args.pixel_stride, args.seed_stride, args.voxel_mm) <= 0:
        parser.error("sampling intervals must be positive")
    if args.frame_stride < 1 or (args.frame_stride != 1 and not args.annotations_only):
        parser.error("frame-stride must be positive; sparse annotation requires --annotations-only")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    manifest = json.loads((args.input / "manifest.json").read_text())
    if args.episodes is None:
        manifest = manifest[: args.max_episodes]
    else:
        selected = set(args.episodes)
        missing = selected - {item["episode"] for item in manifest}
        if missing:
            parser.error(f"Unknown episode IDs: {sorted(missing)}")
        manifest = [item for item in manifest if item["episode"] in selected]
    report = {"settings": vars(args) | {"input": str(args.input), "output": str(args.output)}, "episodes": []}
    if args.resume and (args.output / "report.json").exists():
        report["episodes"] = json.loads((args.output / "report.json").read_text())["episodes"]
    completed = {row["episode"] for row in report["episodes"]}

    for item in manifest:
        episode_id = item["episode"]
        if episode_id in completed and all(
            (args.output / f"episode_{episode_id:05d}_{camera}_visible_surface.npz").exists()
            for camera in CAMERAS
        ):
            continue
        episode = args.input / f"episode_{episode_id:05d}"
        metadata = json.loads((episode / "metadata.json").read_text())
        frames = {camera: read_frames(episode / f"{camera}.mp4") for camera in CAMERAS}
        height, width = frames[CAMERAS[0]][0].shape[:2]
        with np.load(episode / "sim_state.npz") as state:
            count = len(state["qpos"])
            frame_indices = np.arange(0, count, args.frame_stride)
            if any(len(video) != count for video in frames.values()):
                raise ValueError("RGB and simulator state frame counts differ")
            suite = benchmark.get_benchmark_dict()[metadata["suite"]]()
            task = suite.get_task(int(metadata["task_id"]))
            bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_names=list(CAMERAS), camera_heights=height, camera_widths=width,
                                     camera_depths=True, control_freq=int(metadata["fps"]), horizon=1000, ignore_done=True)
            episode_report = {"episode": episode_id, "task": item["task"], "cameras": {}}
            try:
                env.reset()
                sim = env.env.sim
                if list(sim.model.body_names) != metadata["body_names"]:
                    raise ValueError("Recorded body names do not match the reconstructed simulator")
                pose_error = restore_fixed_body_poses(sim, state)
                if pose_error > 1e-5:
                    raise ValueError(f"Body poses do not match the recorded simulation: {pose_error} m")
                episode_report["max_body_pose_error_m"] = pose_error
                colors, labels = select_subjects(sim, metadata["body_names"], args.include_objects)
                episode_report["subjects"] = {labels[bid]: list(color) for bid, color in colors.items()}
                buffers = {camera: {} for camera in CAMERAS}
                alignment = {camera: [] for camera in CAMERAS}
                local_points, bids, gids, seed_frames, seed_cameras = [], [], [], [], []
                seen = set()
                seeds = (set(range(0, count, args.seed_stride)) | {int(frame_indices[-1])}) & set(frame_indices.tolist())
                for t in frame_indices:
                    restore(sim, state, t)
                    for camera_id, camera in enumerate(CAMERAS):
                        rgb, depth, geom_ids = render_buffers(sim, camera, height, width)
                        buffers[camera][t] = (depth, geom_ids)
                        alignment[camera].append(float(np.abs(rgb.astype(float) - frames[camera][t].astype(float)).mean()))
                        if t not in seeds:
                            continue
                        local, body, geom, cells = surface_samples(sim, depth, geom_ids, camera, colors, args.pixel_stride, args.voxel_mm / 1000)
                        for p, b, g, key in zip(local, body, geom, cells, strict=True):
                            key = tuple(key)
                            if key in seen:
                                continue
                            seen.add(key)
                            local_points.append(p)
                            bids.append(b)
                            gids.append(g)
                            seed_frames.append(t)
                            seed_cameras.append(camera_id)
                local_points = np.asarray(local_points, dtype=np.float64).reshape(-1, 3)
                bids, gids = np.asarray(bids, dtype=np.int32), np.asarray(gids, dtype=np.int32)
                world_tracks = np.full((count, len(bids), 3), np.nan, dtype=np.float32)
                tracks = {camera: {key: [] for key in ("xy", "visible", "in_frame", "camera_z")} for camera in CAMERAS}
                for t in frame_indices:
                    restore(sim, state, t)
                    rotations = sim.data.body_xmat[bids].reshape(-1, 3, 3)
                    world = np.einsum("nij,nj->ni", rotations, local_points) + sim.data.body_xpos[bids]
                    world_tracks[t] = world.astype(np.float32)
                    for camera in CAMERAS:
                        depth, geom_ids = buffers[camera][t]
                        values = project_visible(sim, camera, world, gids, depth, geom_ids, args.depth_atol_mm / 1000, args.depth_rtol)
                        for key, value in zip(tracks[camera], values, strict=True):
                            tracks[camera][key].append(value)
            finally:
                env.close()
            point_count = len(local_points)
            rendered_views = {}
            if not args.annotations_only:
                save_legend(colors, labels, args.output / f"episode_{episode_id:05d}_legend.png")
            point_labels = np.asarray([labels[int(bid)] for bid in bids])
            for camera_id, camera in enumerate(CAMERAS):
                data = {}
                for key, values in tracks[camera].items():
                    sparse = np.stack(values)
                    fill = False if sparse.dtype == bool else np.nan
                    data[key] = np.full((count, *sparse.shape[1:]), fill, dtype=sparse.dtype)
                    data[key][frame_indices] = sparse
                occluded = data["in_frame"] & ~data["visible"]
                filename = f"episode_{episode_id:05d}_{camera}_visible_surface"
                np.savez_compressed(args.output / f"{filename}.npz", **data, occluded=occluded, world_xyz=world_tracks,
                                    body_id=bids, geom_id=gids, local_xyz=local_points.astype(np.float32), seed_frame=np.asarray(seed_frames),
                                    seed_camera=np.asarray(seed_cameras), camera_names=np.asarray(CAMERAS),
                                    point_label=point_labels, body_names=np.asarray(metadata["body_names"]),
                                    annotated_frame_indices=frame_indices)
                if not args.annotations_only:
                    rendered = overlay(frames[camera], data["xy"], data["visible"], bids, colors)
                    rendered_views[camera] = rendered
                    write_video(args.output / f"{filename}.mp4", rendered, int(metadata["fps"]))
                    contact_sheet(rendered, args.output / f"{filename}.jpg")
                anchor = np.asarray(seed_cameras) == camera_id
                anchor_indices = np.flatnonzero(anchor)
                anchor_visible = data["visible"][np.asarray(seed_frames)[anchor], anchor_indices]
                stats = {
                    "points": point_count, "frames": count, "annotated_frames": len(frame_indices),
                    "mean_in_frame": float(data["in_frame"][frame_indices].sum(axis=1).mean()),
                    "mean_visible": float(data["visible"][frame_indices].sum(axis=1).mean()),
                    "mean_occluded_in_frame": float(occluded[frame_indices].sum(axis=1).mean()),
                    "visible_at_own_seed_fraction": float(anchor_visible.mean()) if len(anchor_visible) else None,
                    "rgb_replay_mae_0_255": float(np.mean(alignment[camera])),
                    "mean_visible_per_subject": {
                        name: float(data["visible"][frame_indices][:, point_labels == name].sum(axis=1).mean())
                        for name in dict.fromkeys(labels.values())
                    },
                }
                episode_report["cameras"][camera] = stats
            if not args.annotations_only:
                paired = paired_views(rendered_views)
                paired_name = f"episode_{episode_id:05d}_two_views_visible_surface"
                write_video(args.output / f"{paired_name}.mp4", paired, int(metadata["fps"]))
                contact_sheet(paired, args.output / f"{paired_name}.jpg")
            report["episodes"] = [row for row in report["episodes"] if row["episode"] != episode_id]
            report["episodes"].append(episode_report)
            print(json.dumps(episode_report), flush=True)
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
