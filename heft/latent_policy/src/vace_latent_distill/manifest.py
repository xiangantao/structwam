"""Index all LIBERO training episodes/cameras; sample windows at training time."""

import argparse
import hashlib
import json
from pathlib import Path


def build(root, output):
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    root, output = Path(root).resolve(), Path(output).resolve()
    info = json.loads((root / "meta/info.json").read_text())
    if info["codebase_version"] != "v3.0":
        raise ValueError("Expected local LeRobot v3 metadata")
    episodes = ds.dataset(str(root / "meta/episodes"), format="parquet").to_table().to_pylist()
    frame_table = ds.dataset(str(root / "data"), format="parquet").to_table(
        columns=["episode_index", "task_index", "frame_index"]
    )
    summary = (
        frame_table.group_by("episode_index")
        .aggregate(
            [
                ("task_index", "min"),
                ("task_index", "max"),
                ("frame_index", "min"),
                ("frame_index", "max"),
                ("frame_index", "count"),
            ]
        )
        .to_pylist()
    )
    stats = {row["episode_index"]: row for row in summary}
    tasks = {
        int(row["task_index"]): row["__index_level_0__"]
        for row in pq.read_table(root / "meta/tasks.parquet").to_pylist()
    }
    first, last = map(int, info["splits"]["train"].split(":"))
    rows, fps = [], float(info["fps"])
    cameras = ("observation.images.image", "observation.images.image2")
    for episode in sorted(episodes, key=lambda row: row["episode_index"]):
        eid, length = int(episode["episode_index"]), int(episode["length"])
        if not first <= eid < last:
            continue
        summary = stats[eid]
        if (
            summary["task_index_min"] != summary["task_index_max"]
            or summary["frame_index_min"] != 0
            or summary["frame_index_max"] != length - 1
            or summary["frame_index_count"] != length
        ):
            raise ValueError(f"Invalid episode metadata: {eid}")
        if length < 49:
            raise ValueError(f"Training episode {eid} has fewer than49 frames")
        task = int(summary["task_index_min"])
        for camera in cameras:
            key = f"videos/{camera}"
            video = root / info["video_path"].format(
                video_key=camera,
                chunk_index=int(episode[key + "/chunk_index"]),
                file_index=int(episode[key + "/file_index"]),
            )
            begin = float(episode[key + "/from_timestamp"]) * fps
            end = float(episode[key + "/to_timestamp"]) * fps
            if abs(begin - round(begin)) > 1e-3 or abs(end - round(begin) - length) > 1e-3:
                raise ValueError("Nonintegral episode timestamp offset")
            if not video.is_file():
                raise FileNotFoundError(video)
            rows.append(
                {
                    "id": f"e{eid:04d}-{camera.rsplit('.', 1)[-1]}",
                    "episode": eid,
                    "task_index": task,
                    "task": tasks[task],
                    "camera": camera,
                    "video": str(video),
                    "fps": fps,
                    "episode_start_frame": round(begin),
                    "length": length,
                }
            )
    if len(rows) != 2 * (last - first):
        raise ValueError("Manifest must cover every training episode and both cameras")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    audit = {
        "schema": "vace-latent-episodes-v1",
        "episodes": len(rows) // 2,
        "rows": len(rows),
        "frames": int(info["total_frames"]),
        "tasks": len({row["task_index"] for row in rows}),
        "cameras": list(cameras),
        "observed": 33,
        "future": 16,
        "fps": fps,
        "sampling": "Random49-frame windows reseeded each epoch; no feature cache by default",
        "manifest_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".json").write_text(json.dumps(audit, indent=2))
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.output), indent=2))


if __name__ == "__main__":
    main()
