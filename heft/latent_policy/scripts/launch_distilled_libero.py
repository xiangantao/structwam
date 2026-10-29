"""Prepare and preflight exactly one local distilled LIBERO suite.

This helper never trains, downloads, stages, copies, or launches a suite list.
The required ``--suite`` selects one independent output config and run directory.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
WORKSPACE = Path(__file__).resolve().parents[3]
POLICY_ROOT = WORKSPACE / "heft" / "latent_policy"


def _default_runtime() -> Path:
    configured = os.environ.get("LATENT_DISTILLED_PYTHON")
    return Path(configured) if configured else WORKSPACE / "heft" / ".venv" / "bin" / "python"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--dataset-root", type=Path, default=WORKSPACE / "datasets" / "libero")
    parser.add_argument(
        "--template", type=Path, default=POLICY_ROOT / "configs" / "libero_distilled_template.json"
    )
    parser.add_argument(
        "--output", type=Path, help="Config path; defaults to configs/generated/distilled/<suite>.json"
    )
    parser.add_argument(
        "--run-dir", type=Path, help="Independent ACT output directory; defaults to runs/<suite>_distilled"
    )
    parser.add_argument("--distilled-root", type=Path, default=WORKSPACE / "heft-distill")
    parser.add_argument(
        "--distilled-checkpoint",
        type=Path,
        default=WORKSPACE / "runs" / "wan_jepa_distillation_libero_equal6" / "step_0007560.pt",
    )
    parser.add_argument("--vjepa-root", type=Path, default=WORKSPACE / "vjepa2")
    parser.add_argument("--distilled-runtime-python", type=Path, default=_default_runtime())
    parser.add_argument("--task-manifest", type=Path)
    parser.add_argument("--batch-size", type=int, default=8, help="Per-process ACT batch (default: 8)")
    parser.add_argument("--world-size", type=int, default=8, help="dp_replicate degree (default: 8)")
    parser.add_argument(
        "--sample-budget", type=int, default=800_000, help="Sample exposures used to derive steps"
    )
    parser.add_argument(
        "--skip-preflight", action="store_true", help="Only skip the read-only dependency/path check"
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    output = args.output or POLICY_ROOT / "configs" / "generated" / "distilled" / f"{args.suite}.json"
    run_dir = args.run_dir or WORKSPACE / "runs" / f"{args.suite}_distilled"
    # Preserve the venv wrapper; resolving its symlink would drop site-packages.
    python = Path(os.environ.get("LATENT_POLICY_PYTHON", sys.executable))
    env = os.environ.copy()
    source_paths = [
        POLICY_ROOT / "src",
        WORKSPACE / "heft",
        WORKSPACE / "heft" / "src",
        WORKSPACE / "repos" / "lerobot" / "src",
    ]
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in source_paths) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.update(
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        WANDB_MODE="offline",
    )
    prepare = [
        str(python),
        str(POLICY_ROOT / "scripts" / "run.py"),
        "prepare",
        "--dataset-root",
        str(args.dataset_root),
        "--suite",
        args.suite,
        "--encoder",
        "distilled",
        "--template",
        str(args.template),
        "--output",
        str(output),
        "--run-dir",
        str(run_dir),
        "--distilled-root",
        str(args.distilled_root),
        "--distilled-checkpoint",
        str(args.distilled_checkpoint),
        "--distilled-runtime-python",
        str(args.distilled_runtime_python),
        "--vjepa-root",
        str(args.vjepa_root),
        "--batch-size",
        str(args.batch_size),
        "--world-size",
        str(args.world_size),
        "--sample-budget",
        str(args.sample_budget),
    ]
    if args.task_manifest:
        prepare.extend(("--task-manifest", str(args.task_manifest)))
    subprocess.run(prepare, cwd=WORKSPACE, env=env, check=True)
    if not args.skip_preflight:
        subprocess.run(
            [str(python), str(POLICY_ROOT / "scripts" / "preflight.py"), "--config", str(output)],
            cwd=WORKSPACE,
            env=env,
            check=True,
        )
    print(
        json.dumps(
            {
                "suite": args.suite,
                "config": str(output.resolve()),
                "audit": str(output.with_suffix(".audit.json").resolve()),
                "run_dir": str(run_dir.resolve()),
                "world_size": args.world_size,
                "global_batch": args.world_size * args.batch_size,
                "sample_budget": args.sample_budget,
                "training_launched": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
