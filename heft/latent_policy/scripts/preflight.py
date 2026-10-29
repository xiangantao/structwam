"""Read-only, stdlib-only environment check. Does not import encoders or touch GPUs."""

import argparse
import ast
import importlib.metadata
import json
import os
import re
import subprocess
import tomllib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    versions, issues = {}, []
    for name in (
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "draccus",
        "datasets",
        "gymnasium",
        "safetensors",
        "av",
        "pyarrow",
        "deepdiff",
        "jsonlines",
        "termcolor",
        "wandb",
        "hf-libero",
        "robosuite",
        "mujoco",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
            issues.append(f"Missing distribution: {name}")
    act_path = root / "repos/lerobot/src/lerobot/policies/act/modeling_act.py"
    if act_path.is_file():
        tree = ast.parse(act_path.read_text())
        methods = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        for name in ("_make_model", "_build_visual_backbone", "_visual_tokens"):
            if name not in methods:
                issues.append(f"LeRobot ACT extension missing: {name}")
    else:
        issues.append(f"Missing local LeRobot tree: {act_path}")
    constraints = {}
    pyproject = root / "repos/lerobot/pyproject.toml"
    if pyproject.is_file():
        try:
            from packaging.requirements import Requirement
        except ImportError:
            issues.append("Missing packaging; unable to check declared LeRobot version constraints")
        else:
            project = tomllib.loads(pyproject.read_text())["project"]
            for spec in project["dependencies"]:
                requirement = Requirement(spec)
                if requirement.marker and not requirement.marker.evaluate():
                    continue
                try:
                    version = importlib.metadata.version(requirement.name)
                except importlib.metadata.PackageNotFoundError:
                    version = None
                constraints[requirement.name] = {
                    "installed": version,
                    "required": str(requirement.specifier),
                }
                if version is None or version not in requirement.specifier:
                    issues.append(
                        f"LeRobot dependency constraint: {requirement}; installed={version}"
                    )
    config = None
    runtime_name = "wan"
    if args.config:
        config = json.loads(args.config.read_text())
        policy, dataset = config["policy"], config["dataset"]
        encoder = policy.get("encoder")
        if len(policy.get("task_names", [])) != 10 or not dataset.get("episodes"):
            issues.append("Generate a suite-specific config; the template is not trainable")
        if encoder == "wan":
            fields = ("wan_model_path", "channel_mask")
        elif encoder == "vjepa2":
            fields = ("vjepa_root", "vjepa_checkpoint")
        elif encoder == "wan_jepa":
            fields = ("wan_model_path", "channel_mask", "vjepa_root", "jepa_model_path", "artifact_dir")
        elif encoder == "distilled":
            runtime_name = "distilled"
            fields = ("distilled_root", "distilled_checkpoint", "vjepa_root")
            if policy.get("dim_model") != 256 or policy.get("n_heads") != 8:
                issues.append("Distilled suite requires ACT dim_model=256 and n_heads=8")
            checksum = policy.get("distilled_checkpoint_sha256", "")
            if not re.fullmatch(r"[0-9a-fA-F]{64}", checksum):
                issues.append("distilled_checkpoint_sha256 must be a 64-character hex digest")
            runtime = policy.get("distilled_runtime_python", "")
            if not runtime or not Path(runtime).is_file():
                issues.append("Missing local policy path: distilled_runtime_python")
        else:
            issues.append(f"Unknown frozen encoder: {encoder!r}")
            fields = ()
        for key in fields:
            value = policy.get(key)
            path = Path(value) if value else None
            if path is None or not path.exists():
                issues.append(f"Missing local policy path: {key}")
            elif key.endswith("root") and not path.is_dir():
                issues.append(f"Policy root is not a directory: {key}")
            elif key.endswith("checkpoint") or key in ("wan_model_path", "channel_mask"):
                if not path.is_file() and key not in ("wan_model_path",):
                    issues.append(f"Policy checkpoint/mask is not a file: {key}")
        if not dataset.get("root") or not (Path(dataset["root"]) / "meta/info.json").is_file():
            issues.append("Local LeRobot dataset metadata is missing")
        if config.get("env", {}).get("task") != policy.get("suite"):
            issues.append("Training suite and evaluation suite differ")
        if config.get("env", {}).get("max_parallel_tasks", 1) != 1:
            issues.append("Stateful policy evaluation requires max_parallel_tasks=1")
    runtime_python = os.environ.get("LATENT_WAN_PYTHON", str(root / "heft/.venv/bin/python"))
    if config is not None:
        if runtime_name == "distilled":
            runtime_python = config["policy"].get("distilled_runtime_python", runtime_python)
        elif config["policy"].get("wan_runtime_python"):
            runtime_python = config["policy"]["wan_runtime_python"]
    runtime_versions = {}
    runtime_packages = ["torch", "transformers"]
    if runtime_name == "wan":
        runtime_packages.append("diffusers")
    try:
        probe_code = (
            "import importlib.metadata as m, importlib.util as u, json; "
            f"names={runtime_packages!r}; "
            "print(json.dumps({n:(m.version(n) if u.find_spec(n) else None) for n in names}))"
        )
        probe = subprocess.run(
            [runtime_python, "-c", probe_code],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        runtime_versions = json.loads(probe.stdout)
        for name, version in runtime_versions.items():
            # Wan deliberately uses the checked-out modified Diffusers source tree,
            # so it is not expected to appear in importlib.metadata.
            if runtime_name == "wan" and name == "diffusers" and (root / "heft/diffusers/src/diffusers").is_dir():
                continue
            if version is None:
                issues.append(f"{runtime_name.capitalize()} runtime missing distribution: {name}")
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        issues.append(f"{runtime_name.capitalize()} runtime probe failed: {exc}")
    print(
        json.dumps(
            {
                "ready": not issues,
                "versions": versions,
                "lerobot_constraints": constraints,
                "issues": issues,
                "runtime": {"name": runtime_name, "python": runtime_python, "versions": runtime_versions},
                "wan_runtime": {"python": runtime_python, "versions": runtime_versions},
            },
            indent=2,
        )
    )
    raise SystemExit(0 if not issues else 1)


if __name__ == "__main__":
    main()
