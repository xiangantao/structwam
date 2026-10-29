"""Launched with the existing Wan Python, without importing the LeRobot plugin package."""

import importlib
import json
import os
import sys
import time
import traceback
import types
from multiprocessing.connection import Client
from pathlib import Path


def main():
    startup = json.loads(sys.stdin.readline())
    connection = Client(tuple(startup["address"]), authkey=bytes.fromhex(startup["key"]))
    encoder = None
    try:
        root = Path(__file__).resolve().parents[1]
        heft = root.parent
        sys.path[:0] = [str(heft / "diffusers/src"), str(heft / "src"), str(heft)]
        for variable in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
            os.environ[variable] = "1"
        package = types.ModuleType("wan_adapter_runtime")
        package.__path__ = [str(root / "src/lerobot_policy_latent_act/encoders")]
        sys.modules[package.__name__] = package
        module = importlib.import_module("wan_adapter_runtime.wan")
        import torch

        torch.set_num_threads(2)
        while True:
            request = connection.recv()
            if request[0] == "close":
                break
            if request[0] == "init":
                encoder = module.WanEncoder(types.SimpleNamespace(**request[1]), request[2])
                connection.send(("ok", "ready"))
            elif request[0] in ("encode", "encode_profiled"):
                profiled = request[0] == "encode_profiled"
                if profiled:
                    torch.cuda.synchronize(encoder.device)
                    torch.cuda.reset_peak_memory_stats(encoder.device)
                started = time.perf_counter()
                result = encoder.encode([torch.from_numpy(array) for array in request[1]])
                if profiled:
                    torch.cuda.synchronize(encoder.device)
                encoded = time.perf_counter()
                # NumPy payloads avoid PyTorch's shared-memory file descriptor service.
                # Float32 represents every BF16/FP16 value exactly during transport.
                payload = [
                    (
                        features.cpu().float().numpy(),
                        str(features.dtype).removeprefix("torch."),
                    )
                    for features in result
                ]
                if profiled:
                    payload = {
                        "features": payload,
                        "encode_s": encoded - started,
                        "pack_s": time.perf_counter() - encoded,
                        "peak_allocated_gib": torch.cuda.max_memory_allocated(encoder.device)
                        / 2**30,
                    }
                connection.send(("ok", payload))
            else:
                raise ValueError(f"Unknown worker request: {request[0]}")
    except (EOFError, BrokenPipeError):
        pass
    except Exception:  # noqa: BLE001 -- propagate the complete worker failure to its owner
        connection.send(("error", traceback.format_exc()))
    finally:
        if encoder is not None:
            encoder.close()
        connection.close()


if __name__ == "__main__":
    main()
