"""Check the actual node runtime without loading an encoder or starting training."""

import argparse
import importlib
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

from ...upstream import load_upstream
from .config import RunConfig
from .train import preflight


def check_filesystems(output, temporary):
    output.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vjepa-check-", dir=output) as directory:
        directory = Path(directory)
        source = directory / "source"
        source.write_bytes(b"checkpoint-io-check")
        os.replace(source, directory / "epoch")
        os.link(directory / "epoch", directory / "latest")
        if (directory / "latest").read_bytes() != b"checkpoint-io-check":
            raise RuntimeError("Checkpoint atomic rename/hardlink check failed")
    with (
        tempfile.TemporaryDirectory(prefix="vjepa-socket-", dir=temporary) as directory,
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection,
    ):
        connection.bind(str(Path(directory) / "socket"))


def check(config, *, cpu_only=False, expected_gpus=8):
    issues = preflight(config)
    report = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "environment_prefix": sys.prefix,
        "uid": os.geteuid(),
        "torch": torch.__version__,
        "torch_file": torch.__file__,
        "torch_cuda": torch.version.cuda,
        "cuda_checked": not cpu_only,
        "packages": {},
        "compiler_tools": {},
    }
    for name in ("torchvision", "timm", "decord", "pandas", "einops", "wandb"):
        try:
            importlib.import_module(name)
            report["packages"][name] = importlib.metadata.version(name)
        except (ImportError, OSError, RuntimeError) as error:
            issues.append(f"Cannot import {name}: {error}")
    if not issues:
        try:
            load_upstream(config.path(config.vjepa_root))
            importlib.import_module("src.models.vision_transformer")
            check_filesystems(
                config.path(config.output_dir), Path(tempfile.gettempdir())
            )
            report["filesystem_checks"] = (
                "atomic rename, hardlink, local Unix socket passed"
            )
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            issues.append(str(error))
    # The baseline is eager; these tools are reported to catch the historical
    # missing-execute-bit problem, but are not prerequisites for this runner.
    try:
        triton = importlib.import_module("triton")
        directory = Path(triton.__file__).parent / "backends/nvidia/bin"
        for name in ("ptxas", "ptxas-blackwell", "cuobjdump", "nvdisasm"):
            path = directory / name
            try:
                result = subprocess.run(
                    [str(path), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                report["compiler_tools"][name] = {
                    "executable": os.access(path, os.X_OK),
                    "exit_code": result.returncode,
                }
            except (OSError, subprocess.TimeoutExpired) as error:
                report["compiler_tools"][name] = {"error": str(error)}
    except ImportError:
        report["compiler_tools"]["triton"] = (
            "not installed; eager baseline does not require compilation"
        )
    if not cpu_only:
        count = torch.cuda.device_count()
        report["visible_gpus"] = count
        if count != expected_gpus:
            issues.append(
                f"Expected {expected_gpus} visible GPUs, found {count}; use one instance with eight GPUs"
            )
        else:
            try:
                names = []
                for index in range(count):
                    with torch.cuda.device(index):
                        a = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
                        if not torch.equal(a @ a, torch.full_like(a, 32)):
                            raise RuntimeError(
                                f"BF16 CUDA matrix check failed on GPU {index}"
                            )
                        torch.cuda.synchronize(index)
                        names.append(torch.cuda.get_device_name(index))
                        del a
                report["gpu_names"] = names
                report["cuda_check"] = (
                    "BF16 matrix multiply passed on every visible GPU"
                )
            except RuntimeError as error:
                issues.append(f"CUDA execution failed: {error}")
    report.update(ready=not issues, issues=issues)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    config = replace(
        RunConfig.load(args.config),
        model_path=str(args.model_path.resolve()),
        output_dir=str(args.output_dir.resolve()),
        extract_lanes=2,
    )
    report = check(config, cpu_only=args.cpu_only)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
