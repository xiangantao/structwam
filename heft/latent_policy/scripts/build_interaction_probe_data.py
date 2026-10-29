#!/usr/bin/env python3
"""Build small, contiguous LIBERO wrist interaction windows.

The source replays are 20 Hz.  We keep a native-rate sequence of 49 real
frames: 32 frames up to and including the event, followed by 16 outcome
frames.  The script only packages existing replay data; it does not perturb
actions or claim that heuristic labels are ground-truth event annotations.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import av
import numpy as np


WINDOW_BEFORE = 32
WINDOW_AFTER = 16
SOURCE_FPS = 20
OUTPUT_FPS = 20
GRIPPER_BODY_NAMES = ("gripper0_eef", "robot0_right_hand")


def read_video(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "16", "preset": "medium"}
        for image in frames:
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def canonical(text: str) -> str:
    text = re.sub(r"[_-]+", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def target_phrase(task: str) -> str:
    text = canonical(task)
    match = re.search(r"\b(?:pick up|put|place|release) the (.+)$", text)
    if match is None:
        match = re.search(r"\b(?:grasp|move) the (.+)$", text)
    if match is None:
        return ""
    phrase = match.group(1)
    phrase = re.split(
        r"\s+(?:between|next to|from|on|into|in|to|and place|and put|and release)\b",
        phrase,
        maxsplit=1,
    )[0]
    return phrase.strip()


def find_target_body(body_names: list[str], task: str) -> int | None:
    phrase = target_phrase(task)
    if not phrase:
        return None
    phrase_tokens = set(phrase.split())
    candidates: list[tuple[int, int, str]] = []
    for index, name in enumerate(body_names):
        if not name.endswith("_main"):
            continue
        stem = canonical(name.removesuffix("_main"))
        tokens = set(stem.split())
        score = len(tokens & phrase_tokens)
        if score:
            # Prefer an object whose full phrase occurs in the task, then _1.
            exact = int(phrase in stem)
            first_instance = int(stem.endswith(" 1"))
            candidates.append((score * 10 + exact * 3 + first_instance, index, name))
    if not candidates:
        return None
    return max(candidates)[1]


def event_indices(actions: np.ndarray, sign: int) -> list[int]:
    """Return state-frame indices after a gripper command changes sign."""
    command = actions[:, -1]
    if sign > 0:
        transitions = (command[:-1] <= 0) & (command[1:] > 0)
    else:
        transitions = (command[:-1] > 0) & (command[1:] <= 0)
    # action[t] is applied between state t and state t+1; command[t+1] is the
    # first command at the new state, so use t+1 as the event state index.
    return (np.flatnonzero(transitions) + 1).tolist()


def classify_grasp(
    body_xpos: np.ndarray,
    target_id: int,
    hand_id: int,
    event: int,
    state_indices: np.ndarray,
) -> tuple[int, str, dict[str, float]]:
    target_z = body_xpos[:, target_id, 2]
    hand = body_xpos[:, hand_id]
    target = body_xpos[:, target_id]
    before = target_z[max(0, event - 8) : event + 1]
    after = target_z[event : min(len(target_z), event + 33)]
    baseline = float(np.median(before)) if len(before) else float(target_z[event])
    lift = float(np.max(after) - baseline) if len(after) else 0.0
    distance = np.linalg.norm(target - hand, axis=1)
    contact_distance = float(np.min(distance[event : min(len(distance), event + 9)]))
    tail = distance[event : min(len(distance), event + 17)]
    held_distance = float(np.median(tail[-5:])) if len(tail) else contact_distance
    success = int(lift >= 0.02 and held_distance <= 0.10)
    if success:
        failure_type = "none"
    elif contact_distance > 0.08:
        failure_type = "no_contact"
    elif lift < 0.02:
        failure_type = "not_lifted"
    else:
        failure_type = "slip"
    margins = {
        "lift_m": lift,
        "contact_distance_m": contact_distance,
        "held_distance_m": held_distance,
        "state_index": float(event),
    }
    return success, failure_type, margins


def make_window(
    frames: list[np.ndarray],
    state: dict[str, np.ndarray],
    event: int,
    frame_stride: int,
    output_fps: int,
) -> tuple[list[np.ndarray], np.ndarray] | None:
    start = event - frame_stride * WINDOW_BEFORE
    stop = event + frame_stride * WINDOW_AFTER
    if start < 0 or stop >= len(frames):
        return None
    source_indices = np.arange(start, stop + 1, frame_stride, dtype=np.int64)
    if len(source_indices) != WINDOW_BEFORE + WINDOW_AFTER + 1:
        raise AssertionError(source_indices)
    selected = [frames[int(i)] for i in source_indices]
    return selected, source_indices


def task_success_at(metadata: dict, state: dict[str, np.ndarray], frame: int) -> bool:
    """Evaluate the saved simulator state against the native LIBERO predicate."""
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from collect_libero_long_videos import make_env

    _, _, env, init_states = make_env(
        metadata["suite"],
        int(metadata["task_id"]),
        64,
        64,
        int(metadata["fps"]),
        len(state["qpos"]),
    )
    try:
        env.reset()
        env.set_init_state(init_states[int(metadata["init_state_id"])])
        sim = env.env.sim
        sim.data.qpos[:] = state["qpos"][frame]
        sim.data.qvel[:] = state["qvel"][frame]
        sim.forward()
        return bool(env.check_success())
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--max-episodes", type=int, default=200)
    parser.add_argument("--include-place", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict] = []
    skipped: list[dict] = []
    task_success_cache: dict[str, tuple[bool, bool]] = {}

    episode_dirs = sorted(source.glob("episode_*/"))[: args.max_episodes]
    for episode_dir in episode_dirs:
        if len(records) >= args.max_samples:
            break
        metadata_path = episode_dir / "metadata.json"
        state_path = episode_dir / "sim_state.npz"
        video_path = episode_dir / "robot0_eye_in_hand.mp4"
        if not (metadata_path.exists() and state_path.exists() and video_path.exists()):
            continue
        metadata = json.loads(metadata_path.read_text())
        body_names = list(metadata["body_names"])
        target_id = find_target_body(body_names, metadata["task"])
        hand_id = next((body_names.index(name) for name in GRIPPER_BODY_NAMES if name in body_names), None)
        if target_id is None or hand_id is None:
            skipped.append({"episode": episode_dir.name, "reason": "target_or_hand_not_found"})
            continue
        state = {key: value for key, value in np.load(state_path).items()}
        actions = state["actions"]
        frames = read_video(video_path)
        close_events = event_indices(actions, sign=1)
        release_events = event_indices(actions, sign=-1)
        # One grasp and one subsequent release per replay keeps the first
        # pass episode-level rather than counting regrips as independent data.
        valid_close = next(
            (e for e in close_events if make_window(frames, state, e, 1, OUTPUT_FPS) is not None),
            None,
        )
        event_specs = [] if valid_close is None else [("grasp", valid_close)]
        if args.include_place and valid_close is not None:
            valid_release = next(
                (
                    e
                    for e in release_events
                    if e > valid_close and make_window(frames, state, e, 1, OUTPUT_FPS) is not None
                ),
                None,
            )
            if valid_release is not None:
                event_specs.append(("place", valid_release))
        for event_type, event in event_specs:
            if len(records) >= args.max_samples:
                break
            window = make_window(frames, state, event, 1, OUTPUT_FPS)
            if window is None:
                continue
            selected_frames, source_indices = window
            sample_id = f"sample_{len(records):05d}"
            sample_dir = output / sample_id
            sample_dir.mkdir()
            write_video(sample_dir / "wrist.mp4", selected_frames, OUTPUT_FPS)
            state_keys = ("qpos", "qvel", "body_xpos", "body_xmat")
            np.savez_compressed(
                sample_dir / "state.npz",
                **{key: state[key][source_indices] for key in state_keys},
                actions=actions[np.clip(source_indices[:-1], 0, len(actions) - 1)],
                source_frame_indices=source_indices,
                target_body_index=np.asarray(target_id, dtype=np.int64),
                hand_body_index=np.asarray(hand_id, dtype=np.int64),
            )
            if event_type == "grasp":
                label, failure_type, margins = classify_grasp(
                    state["body_xpos"], target_id, hand_id, event, source_indices
                )
            else:
                cache_key = episode_dir.name
                if cache_key not in task_success_cache:
                    window_end = int(source_indices[-1])
                    task_success_cache[cache_key] = (
                        task_success_at(metadata, state, event),
                        task_success_at(metadata, state, window_end),
                    )
                success_at_release, success_final = task_success_cache[cache_key]
                label = int(success_final)
                if success_final:
                    failure_type = "none"
                elif success_at_release:
                    failure_type = "unstable_place"
                else:
                    failure_type = "wrong_release"
                margins = {
                    "success_at_release": float(success_at_release),
                    "success_final": float(success_final),
                }
            record = {
                "sample_id": sample_id,
                "episode": episode_dir.name,
                "suite": metadata["suite"],
                "task_id": metadata["task_id"],
                "task": metadata["task"],
                "event_type": event_type,
                "event_index_in_window": WINDOW_BEFORE,
                "source_event_state_index": int(event),
                "source_frame_indices": source_indices.tolist(),
                "source_fps": SOURCE_FPS,
                "output_fps": OUTPUT_FPS,
                "num_frames": len(selected_frames),
                "target_body": body_names[target_id],
                "hand_body": body_names[hand_id],
                "label": label,
                "label_source": "heuristic_replay_state" if event_type == "grasp" else "simulator_task_predicate",
                "failure_type": failure_type,
                "margins": margins,
            }
            (sample_dir / "metadata.json").write_text(json.dumps(record, indent=2))
            records.append(record)

    manifest = {
        "protocol": {
            "real_frames": 49,
            "frames_to_event_inclusive": 33,
            "frames_after_event": 16,
            "event_index": 32,
            "source_fps": SOURCE_FPS,
            "output_fps": OUTPUT_FPS,
            "wrist_camera": "robot0_eye_in_hand",
            "note": "Existing replay windows; no action perturbation; grasp labels are state heuristics and place labels use LIBERO task predicates.",
        },
        "samples": records,
        "skipped": skipped,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with (output / "samples.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(json.dumps({"output": str(output), "samples": len(records), "skipped": len(skipped)}, indent=2))


if __name__ == "__main__":
    main()
