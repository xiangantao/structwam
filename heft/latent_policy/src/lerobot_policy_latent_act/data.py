"""Suite episode selection and config preparation; LeRobot owns decoding and temporal sampling."""

import argparse
import json
import math
import os
from pathlib import Path

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def canonical_task(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Task descriptions must be nonempty strings")
    return " ".join(text.lower().strip().rstrip(".").split())


def select_suite_episodes(episodes, task_names: list[str]) -> dict:
    names = [canonical_task(name) for name in task_names]
    if len(names) != 10 or len(set(names)) != 10:
        raise ValueError("A suite must have exactly 10 distinct task descriptions")
    counts = {name: {"episodes": 0, "frames": 0} for name in names}
    selected, seen = [], set()
    for episode in episodes:
        index = int(episode["episode_index"])
        if index in seen:
            raise ValueError(f"Duplicate episode_index: {index}")
        seen.add(index)
        tasks = {canonical_task(task) for task in episode["tasks"]}
        matches = tasks.intersection(names)
        if not matches:
            continue
        if len(tasks) != 1:
            raise ValueError(f"Episode {index} has ambiguous/mixed task membership: {tasks}")
        name = next(iter(matches))
        length = int(episode["length"])
        if length < 1:
            raise ValueError(f"Empty episode: {index}")
        selected.append(index)
        counts[name]["episodes"] += 1
        counts[name]["frames"] += length
    missing = [name for name, count in counts.items() if not count["episodes"]]
    if missing:
        raise ValueError(f"No local demonstrations for suite tasks: {missing}")
    return {
        "episodes": sorted(selected),
        "tasks": counts,
        "num_frames": sum(count["frames"] for count in counts.values()),
    }


def official_task_names(suite: str) -> list[str]:
    from libero.libero import benchmark

    tasks = benchmark.get_benchmark_dict()[suite]()
    return [tasks.get_task(i).language for i in range(tasks.get_num_tasks())]


def episodes_with_tasks(meta):
    """Recover task membership from frame columns when v3 episode metadata omits tasks."""
    episodes = list(meta.episodes)
    if all("tasks" in row for row in episodes):
        return episodes
    import pyarrow.dataset as arrow_dataset

    task_names = {int(row.task_index): str(name) for name, row in meta.tasks.iterrows()}
    membership = {}
    table = arrow_dataset.dataset(str(Path(meta.root) / "data"), format="parquet")
    for batch in table.scanner(columns=["episode_index", "task_index"]).to_batches():
        for episode, task in zip(
            batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True
        ):
            membership.setdefault(int(episode), set()).add(task_names[int(task)])
    for row in episodes:
        if row["episode_index"] not in membership:
            raise ValueError(f"Missing frame task assignments for episode {row['episode_index']}")
        row["tasks"] = sorted(membership[row["episode_index"]])
    return episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="HuggingFaceVLA/libero")
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--encoder", choices=("wan", "vjepa2", "wan_jepa", "distilled"), required=True)
    parser.add_argument(
        "--task-manifest",
        type=Path,
        help="Optional JSON mapping suite names to 10 official task descriptions",
    )
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--wan-model-path", type=Path)
    parser.add_argument(
        "--wan-runtime-python",
        type=Path,
        default=Path(os.environ["LATENT_WAN_PYTHON"])
        if os.environ.get("LATENT_WAN_PYTHON")
        else None,
    )
    parser.add_argument("--channel-mask", type=Path)
    parser.add_argument("--vjepa-root", type=Path)
    parser.add_argument("--vjepa-checkpoint", type=Path)
    parser.add_argument("--jepa-model-path", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--distilled-root", type=Path)
    parser.add_argument("--distilled-checkpoint", type=Path)
    parser.add_argument("--distilled-runtime-python", type=Path)
    parser.add_argument("--batch-size", type=int, default=8, help="Per-process batch size")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--sample-budget", type=int, default=800_000)
    args = parser.parse_args()
    if min(args.batch_size, args.world_size, args.sample_budget) < 1:
        parser.error("Batch, world size and sample budget must be positive")
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    from .encoders.common import file_sha256

    root = args.dataset_root.resolve(strict=True)
    if not (root / "meta/info.json").is_file():
        parser.error("Expected a complete local LeRobot dataset with meta/info.json")
    names = (
        json.loads(args.task_manifest.read_text())[args.suite]
        if args.task_manifest
        else official_task_names(args.suite)
    )
    meta = LeRobotDatasetMetadata(args.repo_id, root=root)
    if meta.fps <= 0:
        parser.error("Dataset must declare a positive storage timestamp rate")
    selected = select_suite_episodes(episodes_with_tasks(meta), names)
    config = json.loads(args.template.read_text())
    config["dataset"].update(repo_id=args.repo_id, root=str(root), episodes=selected["episodes"])
    config["policy"].update(encoder=args.encoder, suite=args.suite, task_names=names)
    if args.encoder == "wan":
        if args.wan_model_path is None or args.channel_mask is None:
            parser.error("Wan requires --wan-model-path and --channel-mask")
        config["policy"].update(
            wan_model_path=str(args.wan_model_path.resolve(strict=True)),
            channel_mask=str(args.channel_mask.resolve(strict=True)),
            channel_mask_sha256=file_sha256(args.channel_mask),
            wan_runtime_python=str(args.wan_runtime_python.absolute())
            if args.wan_runtime_python
            else "",
        )
    elif args.encoder == "vjepa2":
        if args.vjepa_root is None or args.vjepa_checkpoint is None:
            parser.error("V-JEPA2 requires --vjepa-root and --vjepa-checkpoint")
        config["policy"].update(
            vjepa_root=str(args.vjepa_root.resolve(strict=True)),
            vjepa_checkpoint=str(args.vjepa_checkpoint.resolve(strict=True)),
            vjepa_checkpoint_sha256=file_sha256(args.vjepa_checkpoint),
        )
    elif args.encoder == "wan_jepa":
        jepa_model = args.jepa_model_path or args.vjepa_checkpoint
        artifact_dir = args.artifact_dir
        if args.wan_model_path is None or args.channel_mask is None:
            parser.error("Wan-JEPA requires --wan-model-path and --channel-mask")
        if args.vjepa_root is None or jepa_model is None or artifact_dir is None:
            parser.error("Wan-JEPA requires --vjepa-root, --jepa-model-path and --artifact-dir")
        config["policy"].update(
            wan_model_path=str(args.wan_model_path.resolve(strict=True)),
            channel_mask=str(args.channel_mask.resolve(strict=True)),
            channel_mask_sha256=file_sha256(args.channel_mask),
            wan_runtime_python=str(args.wan_runtime_python.absolute())
            if args.wan_runtime_python
            else "",
            vjepa_root=str(args.vjepa_root.resolve(strict=True)),
            jepa_model_path=str(jepa_model.resolve(strict=True)),
            artifact_dir=str(artifact_dir.resolve(strict=True)),
        )
    else:
        if args.distilled_root is None or args.distilled_checkpoint is None:
            parser.error("Distilled requires --distilled-root and --distilled-checkpoint")
        if args.vjepa_root is None:
            parser.error("Distilled requires --vjepa-root (the student ViT-L source tree)")
        config["policy"].update(
            vjepa_root=str(args.vjepa_root.resolve(strict=True)),
            distilled_root=str(args.distilled_root.resolve(strict=True)),
            distilled_checkpoint=str(args.distilled_checkpoint.resolve(strict=True)),
            distilled_checkpoint_sha256=file_sha256(args.distilled_checkpoint),
            distilled_runtime_python=(
                str(args.distilled_runtime_python.absolute())
                if args.distilled_runtime_python
                else ""
            ),
        )
    global_batch = args.batch_size * args.world_size
    config.update(
        batch_size=args.batch_size,
        steps=math.ceil(args.sample_budget / global_batch),
        output_dir=str(args.run_dir.resolve()),
        job_name=f"{args.suite}_{args.encoder}",
    )
    config["parallelism"] = {"dp_replicate": args.world_size, "dp_shard": 1}
    config["env"]["task"] = args.suite
    # Storage timestamps address consecutive dataset rows; they do not redefine simulator physics.
    # Keep the official LIBERO 20 Hz control rate and execute one action per control step.
    selected["dataset_storage_fps"] = meta.fps
    selected["env_control_fps"] = config["env"].get("fps", 20)
    selected.update(
        suite=args.suite,
        task_names=names,
        world_size=args.world_size,
        global_batch=global_batch,
        steps=config["steps"],
        sample_exposures=config["steps"] * global_batch,
        approximate_epochs=config["steps"] * global_batch / selected["num_frames"],
    )
    for path in (args.output, args.output.with_suffix(".audit.json")):
        if path.exists():
            parser.error(f"Refusing to overwrite {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    args.output.with_suffix(".audit.json").write_text(json.dumps(selected, indent=2) + "\n")
    print(
        json.dumps(
            {
                "config": str(args.output),
                "episodes": len(selected["episodes"]),
                "frames": selected["num_frames"],
                "steps": config["steps"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
