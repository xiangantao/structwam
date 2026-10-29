"""Persistent process adapter for the distilled student runtime."""

import json
import os
import subprocess
import tempfile
import time
from multiprocessing.connection import Listener
from pathlib import Path

import torch

from .common import checked_local_file, rgb_uint8


class DistilledProcessEncoder:
    """Keep distillation imports and their environment out of the ACT process."""

    def __init__(self, config, device, python):
        self.config = config
        self.device = torch.device(device)
        self.process = None
        self.connection = None
        executable = Path(python).absolute()
        if not executable.is_file():
            raise FileNotFoundError(f"Distilled runtime Python not found: {executable}")
        root = Path(config.distilled_root).resolve(strict=True)
        checkpoint = checked_local_file(
            config.distilled_checkpoint, config.distilled_checkpoint_sha256
        )
        vjepa_root = Path(config.vjepa_root).resolve(strict=True)
        if not (root / "latent_video/distillation/model.py").is_file():
            raise ValueError(f"Not a distillation source tree: {root}")
        if not (vjepa_root / "src/models/vision_transformer.py").is_file():
            raise ValueError(f"Not a V-JEPA2 source tree: {vjepa_root}")

        worker = Path(__file__).resolve().parents[3] / "scripts/distilled_worker.py"
        if not worker.is_file():
            raise FileNotFoundError(f"Distilled worker not found: {worker}")
        log = tempfile.NamedTemporaryFile(
            prefix="distilled-worker-", suffix=".log", delete=False
        )  # noqa: SIM115 -- closed in finally
        self.log_path = Path(log.name)
        key = os.urandom(32)
        try:
            with Listener(("127.0.0.1", 0), authkey=key) as listener:
                listener._listener._socket.settimeout(60)
                self.process = subprocess.Popen(
                    [str(executable), "-u", str(worker)],
                    stdin=subprocess.PIPE,
                    stdout=log,
                    stderr=log,
                    env=os.environ.copy(),
                )
                startup = {"address": listener.address, "key": key.hex()}
                self.process.stdin.write((json.dumps(startup) + "\n").encode())
                self.process.stdin.close()
                self.connection = listener.accept()
            options = {
                "distilled_root": str(root),
                "distilled_checkpoint": str(checkpoint),
                "distilled_checkpoint_sha256": config.distilled_checkpoint_sha256,
                "vjepa_root": str(vjepa_root),
                "extract_batch_size": int(getattr(config, "extract_batch_size", 1)),
            }
            self.connection.send(("init", options, str(self.device)))
            health = self._receive(timeout=1800)
            expected = {
                "training": False,
                "any_requires_grad": False,
                "output_shape": [8, 1152, 16, 16],
            }
            if health != expected:
                raise RuntimeError(f"Invalid distilled worker state: {health}")
        except BaseException:
            self.close()
            raise
        finally:
            log.close()

    def _receive(self, timeout=900):
        if self.connection is None or self.process is None:
            raise RuntimeError("Distilled worker is not running")
        deadline = time.monotonic() + timeout
        while not self.connection.poll(1):
            if self.process.poll() is not None:
                raise RuntimeError(f"Distilled worker exited; see {self.log_path}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Distilled worker timed out; see {self.log_path}")
        status, result = self.connection.recv()
        if status != "ok":
            raise RuntimeError(f"Distilled worker failed: {result}\nLog: {self.log_path}")
        return result

    @torch.no_grad()
    def encode(self, images):
        if not images:
            raise ValueError("At least one camera is required")
        images = rgb_uint8(images, frames=16, size=256)
        self.connection.send(("encode", [image.numpy() for image in images]))
        maps = self._receive()
        if not isinstance(maps, list) or len(maps) != len(images):
            raise ValueError("Distilled worker returned the wrong number of cameras")
        batch_size = images[0].shape[0]
        expected = (batch_size, 8, 1152, 16, 16)
        with torch.inference_mode(False):
            outputs = [
                torch.from_numpy(array)
                .to(device=self.device, dtype=torch.float32)
                .detach()
                .clone()
                for array in maps
            ]
        if any(tuple(features.shape) != expected for features in outputs):
            raise ValueError(
                "Distilled worker returned an unexpected feature shape: "
                f"{[tuple(features.shape) for features in outputs]}"
            )
        return outputs

    def close(self):
        if self.connection is not None:
            try:
                self.connection.send(("close",))
            except (BrokenPipeError, EOFError, OSError):
                pass
            self.connection.close()
            self.connection = None
        if self.process is not None:
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process = None
