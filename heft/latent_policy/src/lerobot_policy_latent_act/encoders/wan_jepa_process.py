"""Direct Wan+V-JEPA2 extraction in the isolated Wan runtime.

The policy process only receives detached feature maps.  This keeps the modified
Diffusers/Transformers stack out of LeRobot while avoiding the distilled student.
"""

import json
import os
import subprocess
import tempfile
import time
from multiprocessing.connection import Listener
from pathlib import Path

import torch

from .common import rgb_uint8


class WanJepaProcessEncoder:
    def __init__(self, config, device, python):
        self.config = config
        self.device = torch.device(device)
        self.process = None
        self.connection = None
        executable = Path(python).absolute()
        if not executable.is_file():
            raise FileNotFoundError(f"Wan-JEPA runtime Python not found: {executable}")
        worker = Path(
            os.environ.get("LATENT_WAN_JEPA_WORKER")
            or Path(__file__).resolve().parents[3] / "scripts" / "wan_jepa_worker.py"
        )
        log = tempfile.NamedTemporaryFile(prefix="wan-jepa-worker-", suffix=".log", delete=False)
        self.log_path = Path(log.name)
        key = os.urandom(32)
        try:
            with Listener(("127.0.0.1", 0), authkey=key) as listener:
                listener._listener._socket.settimeout(120)
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
                name: getattr(config, name)
                for name in (
                    "wan_model_path",
                    "channel_mask",
                    "channel_mask_sha256",
                    "jepa_model_path",
                    "vjepa_root",
                    "artifact_dir",
                    "extract_batch_size",
                    "history_frames",
                    "image_size",
                    "feature_seed",
                )
            }
            self.connection.send(("init", options, str(self.device)))
            self._receive(timeout=1800)
        except BaseException:
            self.close()
            raise
        finally:
            log.close()

    def _receive(self, timeout=1800):
        deadline = time.monotonic() + timeout
        while not self.connection.poll(1):
            if self.process.poll() is not None:
                raise RuntimeError(f"Wan-JEPA worker exited; see {self.log_path}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Wan-JEPA worker timed out; see {self.log_path}")
        status, result = self.connection.recv()
        if status != "ok":
            raise RuntimeError(f"Wan-JEPA worker failed: {result}\nLog: {self.log_path}")
        return result

    @torch.no_grad()
    def encode(self, images, seeds=None):
        images = rgb_uint8(images, self.config.history_frames, self.config.image_size)
        if seeds is None:
            self.connection.send(("encode", [image.numpy() for image in images]))
        else:
            self.connection.send(("encode", [image.numpy() for image in images], [int(s) for s in seeds]))
        maps = self._receive()
        if not isinstance(maps, list) or len(maps) != len(images):
            raise ValueError("Wan-JEPA worker returned the wrong number of cameras")
        batch_size = images[0].shape[0]
        expected = (batch_size, 8, 1152, 16, 16)
        with torch.inference_mode(False):
            outputs = [
                torch.from_numpy(array).to(self.device, dtype=torch.float32).detach().clone()
                for array in maps
            ]
        if any(tuple(features.shape) != expected for features in outputs):
            raise ValueError(
                "Wan-JEPA worker returned an unexpected feature shape: "
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
