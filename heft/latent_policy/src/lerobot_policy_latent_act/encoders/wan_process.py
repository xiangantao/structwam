"""Keep Transformers 4 / modified Diffusers in a separate, persistent local Wan process."""

import json
import os
import subprocess
import tempfile
import time
from multiprocessing.connection import Listener
from pathlib import Path

import torch

from .common import rgb_uint8


class WanProcessEncoder:
    def __init__(self, config, device, python):
        self.config = config
        self.device = torch.device(device)
        self.process = None
        self.connection = None
        executable = Path(python).absolute()
        if not executable.is_file():
            raise FileNotFoundError(f"Wan runtime Python not found: {executable}")
        worker = Path(__file__).resolve().parents[3] / "scripts/wan_worker.py"
        log = tempfile.NamedTemporaryFile(prefix="wan-worker-", suffix=".log", delete=False)  # noqa: SIM115 -- closed in finally
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
                name: getattr(config, name)
                for name in (
                    "wan_model_path",
                    "channel_mask",
                    "channel_mask_sha256",
                    "extract_batch_size",
                    "extract_lanes",
                    "compile_blocks",
                    "feature_seed",
                    "history_frames",
                    "image_size",
                )
            }
            self.connection.send(("init", options, str(self.device)))
            self._receive(timeout=900)
        except BaseException:
            self.close()
            raise
        finally:
            log.close()

    def _receive(self, timeout=900):
        deadline = time.monotonic() + timeout
        while not self.connection.poll(1):
            if self.process.poll() is not None:
                raise RuntimeError(f"Wan worker exited; see {self.log_path}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Wan worker timed out; see {self.log_path}")
        status, result = self.connection.recv()
        if status != "ok":
            raise RuntimeError(f"Wan worker failed: {result}\nLog: {self.log_path}")
        return result

    @torch.no_grad()
    def encode(self, images):
        images = rgb_uint8(images, self.config.history_frames, self.config.image_size)
        self.connection.send(("encode", [image.numpy() for image in images]))
        maps = self._receive()
        with torch.inference_mode(False):
            return [
                torch.from_numpy(array).to(device=self.device, dtype=getattr(torch, dtype)).clone()
                for array, dtype in maps
            ]

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
