#!/usr/bin/env python3
"""Closed-loop LIBERO evaluation of our Diffusion Policies (vanilla DP, or DP + maps / tracks / trunk).

Follows LeRobot's LIBERO conventions (``lerobot.envs.libero``, ``lerobot.processor.env_processor``):
each episode starts from LIBERO's fixed init state for that episode index, then 10 no-op steps
(gripper open) let the objects settle; step limits per suite as LeRobot; camera images are
rotated 180 degrees (flip H and W) and the state is eef position + axis-angle (from the
x, y, z, w quaternion) + gripper qpos.  Actions go to the simulator unnormalized and unchanged.

Policy side, as in training: 2 observation steps (t-1, t; clamped at the episode start), the
chunk's 8 executed actions (t .. t+7), EMA weights, DDIM ``--sampler-steps``.  The maps policy
runs the frozen student on the last 33 frames of both cameras (start-of-episode frames
repeated, like the training windows).

Each process evaluates a share of the (task, init state) pairs (``--shard`` / ``--num-shards``)
and appends one json line per episode.  ``--compress av1:<crf>`` passes the policy's input
frames through the training videos' codec first (saved videos stay raw).  ``--check-alignment`` instead saves the first frame of
task 0 (raw and rotated) next to a training frame of the same task, with the state ranges.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch

if os.environ.get("VACE_NPU", "0") == "1":
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from latent_dp.data import minmax  # noqa: E402

CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")   # dataset image, image2
MAX_STEPS = {"libero_spatial": 280, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
NO_OP = [0, 0, 0, 0, 0, 0, -1]
WINDOW = 33


def quat2axisangle(quat):
    """(x, y, z, w) quaternion -> axis-angle (robosuite / LeRobot)."""
    quat = np.asarray(quat, np.float64).copy()
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = math.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, np.float32)
    return (quat[:3] * 2.0 * math.acos(quat[3]) / den).astype(np.float32)


def observe(raw):
    """Raw LIBERO observation -> (images [2, 3, 256, 256] uint8 rotated 180 deg, state [8])."""
    images = np.stack([np.ascontiguousarray(raw[c][::-1, ::-1]).transpose(2, 0, 1) for c in CAMERAS])
    state = np.concatenate((raw["robot0_eef_pos"], quat2axisangle(raw["robot0_eef_quat"]),
                            raw["robot0_gripper_qpos"])).astype(np.float32)
    return images, state


def compressor(spec):
    """``none`` | ``av1:<crf>`` | ``jpeg:<quality>`` -> function on [2, 3, H, W] uint8 camera images.

    AV1 matches how the LeRobot training videos were stored (libsvtav1, yuv420p; one intra
    frame per call, decoded with libdav1d), so the policy sees training-like frames.
    """
    kind, _, level = spec.partition(":")
    if kind == "none":
        return lambda images: images
    if kind == "jpeg":
        import cv2

        def run(img):
            ok, buf = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(level)])
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1]
    elif kind == "av1":
        from fractions import Fraction

        import av

        def run(img):
            encoder = av.CodecContext.create("libsvtav1", "w")
            encoder.width, encoder.height, encoder.pix_fmt = img.shape[1], img.shape[0], "yuv420p"
            encoder.time_base = Fraction(1, 10)
            encoder.options = {"crf": level, "preset": "8", "svtav1-params": "lp=2"}
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24").reformat(format="yuv420p")
            frame.pts = 0
            packets = list(encoder.encode(frame)) + list(encoder.encode(None))
            decoder = av.CodecContext.create("libdav1d", "r")
            decoded = [f for packet in packets for f in decoder.decode(packet)] + list(decoder.decode(None))
            return decoded[-1].to_ndarray(format="rgb24")
    else:
        raise ValueError(f"unknown --compress {spec}")
    return lambda images: np.stack([run(c.transpose(1, 2, 0)).transpose(2, 0, 1) for c in images])


def make_env(suite, task_id, size=256):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    return OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=size, camera_widths=size), task.language


def unminmax(value, low, high):
    return np.where(high - low > 1e-8, (value + 1) / 2 * (high - low) + low, low)


class Policy:
    """Loads a ``train_dp_rgb.py`` or ``train_dp_maps_online.py`` checkpoint (EMA weights)."""

    def __init__(self, checkpoint, kind, device, student=None, roles=None, sampler_steps=10, map_workers=4, drop=()):
        saved = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        self.stats, self.kind, self.device, self.sampler_steps = saved["stats"], kind, device, sampler_steps
        self.drop = tuple(drop)                                       # maps policy branches zeroed at test time
        imagenet = bool(saved["config"].get("rgb_imagenet", False))     # BatchNorm + ImageNet normalization
        if kind == "rgb":
            from latent_dp.rgb import RgbDiffusionPolicy

            self.policy = RgbDiffusionPolicy(rgb_imagenet=imagenet)
        else:
            from latent_dp.online import OnlineMaps
            from latent_dp.policy_maps import MAP_CHANNELS
            from latent_dp.rgb import MapsDiffusionPolicy

            self.policy = MapsDiffusionPolicy(len(MAP_CHANNELS), saved["config"]["track_token_dim"], branch_dropout=0.0,
                                              rgb_imagenet=imagenet, anchors=bool(saved["config"].get("anchors", False)))
            self.online = OnlineMaps(student, roles, device, map_workers)
            self.online.k_basis = (self.stats["k_mean"], self.stats["k_basis"])
        self.policy.load_state_dict(saved["ema"], strict=True)
        self.policy = self.policy.to(device).eval()
        self.generator = torch.Generator(device=device)

    @torch.no_grad()
    def act(self, frames, states, task, seed):
        """``frames``: list of [2, 3, H, W] uint8 up to now; ``states``: list of [8] -> 8 actions [8, 7]."""
        t = len(frames) - 1
        obs = [max(t - 1, 0), t]
        batch = {"images": torch.from_numpy(np.stack([frames[i] for i in obs])[None]).to(self.device).float().div_(255),
                 "state": torch.from_numpy(minmax(np.stack([states[i] for i in obs])[None], self.stats["state_min"],
                                                  self.stats["state_max"]).astype(np.float32)).to(self.device)}
        if self.kind == "maps":
            window = [frames[max(i, 0)] for i in range(t - WINDOW + 1, t + 1)]
            rgb = torch.from_numpy(np.stack(window).transpose(1, 0, 2, 3, 4)).to(self.device)   # [2, 33, 3, H, W]
            maps, tokens, used, trunk, _ = self.online(rgb, [task, task], [False, True])
            batch.update(maps=torch.from_numpy(maps[None]).to(self.device),
                         tracks=torch.from_numpy(tokens[None]).to(self.device),
                         track_used=torch.from_numpy(used[None]).to(self.device), trunk=trunk[None], drop=self.drop)
        self.generator.manual_seed(seed)
        chunk = self.policy.sample(batch, scheduler="DDIM", steps=self.sampler_steps, generator=self.generator)
        actions = self.policy.executed(chunk)[0].float().cpu().numpy()
        return unminmax(actions, self.stats["action_min"], self.stats["action_max"])


def grasp_metrics(log):
    """Simulator ground truth around the first close command (diagnosis only, never a policy input).

    ``log``: per-step eef position, finger qpos, commanded gripper and target-object position.
    Offsets are eef - object centre in world metres; descent speed is the eef z velocity over
    the 3 steps before the close; ``lifted`` = the object rose 3 cm within 40 steps after it;
    ``finger_gap_after`` = finger opening 10 steps later (~0: closed on nothing).
    """
    eef, fingers, command, obj = (np.asarray(log[k]) for k in ("eef", "fingers", "command", "object"))
    closes = [i for i in range(len(command)) if command[i] > 0 and (i == 0 or command[i - 1] <= 0)]
    out = {"close_events": len(closes), "first_close": closes[0] if closes else None}
    if not closes:
        return out
    c = closes[0]
    later = obj[c:c + 40, 2] - obj[c, 2]
    lift = np.flatnonzero(later > 0.03)
    out.update(offset=[round(float(v), 4) for v in eef[c] - obj[c]],
               descent_speed=round(float((eef[c, 2] - eef[max(c - 3, 0), 2]) / max(min(3, c), 1)), 5),
               lateral_speed=round(float(np.linalg.norm(eef[c, :2] - eef[max(c - 3, 0), :2]) / max(min(3, c), 1)), 5),
               lifted=bool(len(lift)), lift_step=int(c + lift[0]) if len(lift) else None,
               finger_gap_at=round(float(fingers[c, 0] - fingers[c, 1]), 4),
               finger_gap_after=round(float(fingers[min(c + 10, len(fingers) - 1), 0] - fingers[min(c + 10, len(fingers) - 1), 1]), 4),
               reopened=bool(any(command[c + 1:] <= 0)), object_moved_before=round(float(np.linalg.norm(obj[c] - obj[0])), 4))
    return out


def run_episode(env, init_state, policy, task, max_steps, seed, video=None, compress=lambda images: images,
                instrument=False):
    env.reset()
    raw = env.set_init_state(init_state)
    for _ in range(10):                                   # let objects settle (LeRobot num_steps_wait)
        raw, _, _, _ = env.step(NO_OP)
    images, state = observe(raw)
    frames, states = [compress(images)], [state]
    target = env.env.obj_body_id[env.env.obj_of_interest[0]] if instrument else None
    log = {"eef": [], "fingers": [], "command": [], "object": []}
    steps, success, decisions, policy_time = 0, False, 0, 0.0
    while steps < max_steps and not success:
        tick = time.time()
        actions = policy.act(frames, states, task, seed + decisions)
        policy_time += time.time() - tick
        decisions += 1
        for action in actions:
            raw, _, done, _ = env.step(action.tolist())
            steps += 1
            if instrument:
                log["eef"].append(np.array(raw["robot0_eef_pos"]))
                log["fingers"].append(np.array(raw["robot0_gripper_qpos"]))
                log["command"].append(float(action[-1]))
                log["object"].append(np.array(env.env.sim.data.body_xpos[target]))
            success = bool(env.check_success())
            images, state = observe(raw)
            frames.append(compress(images))
            states.append(state)
            if video is not None:
                video.append(np.concatenate((images[0], images[1]), 2).transpose(1, 2, 0))
            if success or done or steps >= max_steps:
                break
    result = {"success": success, "steps": steps, "decisions": decisions, "policy_s": round(policy_time, 2)}
    if instrument:
        result["object"] = env.env.obj_of_interest[0]
        result["grasp"] = grasp_metrics(log)
        result["trace"] = {k: np.round(np.asarray(v), 4).tolist() for k, v in log.items()}
    return result


def check_alignment(args, suite):
    import cv2

    from train_latent_dp_online import load_frames

    env, language = make_env(suite, 0)
    env.seed(0)
    env.reset()
    raw = env.set_init_state(suite.get_task_init_states(0)[0])
    for _ in range(10):
        raw, _, _, _ = env.step(NO_OP)
    images, state = observe(raw)
    frames, names = load_frames(args.dataset_root)
    task_index = [i for i, n in names.items() if n == language]
    print("LIBERO task 0:", language, "| dataset task index:", task_index)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("local/libero", root=str(args.dataset_root), video_backend="pyav")
    first = int(np.flatnonzero(frames["task"] == task_index[0])[0]) if task_index else 0
    item = dataset[first]
    data = [(item[k].numpy() * 255).round().astype(np.uint8) for k in ("observation.images.image", "observation.images.image2")]
    rows = []
    for c in range(2):
        raw_image = raw[CAMERAS[c]].transpose(2, 0, 1)
        rows.append(np.concatenate((raw_image, images[c], data[c]), 2))
    sheet = np.concatenate(rows, 1).transpose(1, 2, 0)
    out = args.output / "alignment.png"
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(np.ascontiguousarray(sheet), cv2.COLOR_RGB2BGR))
    saved = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)["stats"]
    print("columns: raw sim | rotated 180 (what the policy sees) | training frame; rows: agentview, wrist ->", out)
    print("sim state      ", np.round(state, 3).tolist())
    print("dataset state  ", np.round(frames["state"][first], 3).tolist())
    print("train state min", np.round(saved["state_min"], 3).tolist())
    print("train state max", np.round(saved["state_max"], 3).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=sorted(MAX_STEPS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--kind", choices=("rgb", "maps"), required=True)
    parser.add_argument("--student", type=Path, help="frozen VACE student (maps policy)")
    parser.add_argument("--roles", type=Path, help="select_policy_roles.py output (maps policy)")
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/ma-user/work/2337888765/workspace/datasets/libero"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50, help="init states per task (LIBERO standard: 50)")
    parser.add_argument("--tasks", type=int, nargs="*", help="task ids (default: all)")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--sampler-steps", type=int, default=10)
    parser.add_argument("--save-videos", type=int, default=0, help="save the first N episodes of each task as mp4")
    parser.add_argument("--map-workers", type=int, default=2, help="CPU workers for the maps post-processing")
    parser.add_argument("--drop", nargs="*", default=[], choices=("maps", "tracks", "trunk"),
                        help="maps policy: zero these condition branches (trained with branch dropout)")
    parser.add_argument("--instrument", action="store_true",
                        help="log simulator ground truth (eef, fingers, target object) and grasp metrics per episode")
    parser.add_argument("--compress", default="none",
                        help="none | av1:<crf> | jpeg:<q>: compress policy input frames like the training videos")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--check-alignment", action="store_true")
    args = parser.parse_args()
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.suite]()
    if args.check_alignment:
        check_alignment(args, suite)
        return
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    policy = Policy(args.checkpoint, args.kind, device, args.student, args.roles, args.sampler_steps, args.map_workers,
                    args.drop)
    compress = compressor(args.compress)
    tasks = args.tasks if args.tasks else list(range(suite.n_tasks))
    jobs = [(t, e) for t in tasks for e in range(args.episodes)][args.shard::args.num_shards]
    args.output.mkdir(parents=True, exist_ok=True)
    log = args.output / f"episodes_shard{args.shard}.jsonl"
    done = set()
    if log.exists():                                       # resume
        done = {(r["task_id"], r["episode"]) for r in map(json.loads, log.read_text().splitlines()) if r}
    env, env_task = None, None
    for task_id, episode in jobs:
        if (task_id, episode) in done:
            continue
        if env_task != task_id:
            if env is not None:
                env.close()
            env, language = make_env(suite, task_id)
            init_states = suite.get_task_init_states(task_id)
            env_task = task_id
        env.seed(args.seed + episode)
        video = [] if episode < args.save_videos else None
        tick = time.time()
        result = run_episode(env, init_states[episode % len(init_states)], policy, language, MAX_STEPS[args.suite],
                             args.seed * 100000 + task_id * 1000 + episode * 37, video, compress, args.instrument)
        result.update(task_id=task_id, episode=episode, task=language, seconds=round(time.time() - tick, 1))
        with log.open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "trace"}), flush=True)
        if video:
            import cv2

            path = args.output / "videos" / f"task{task_id:02d}_ep{episode:02d}_{'ok' if result['success'] else 'fail'}.mp4"
            path.parent.mkdir(exist_ok=True)
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (video[0].shape[1], video[0].shape[0]))
            for frame in video:
                writer.write(cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR))
            writer.release()
    if env is not None:
        env.close()
    if args.kind == "maps":
        policy.online.close()


if __name__ == "__main__":
    main()
