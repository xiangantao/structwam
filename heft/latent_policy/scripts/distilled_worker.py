"""Run the distilled student without importing the ACT environment."""

import atexit
import importlib
import json
import os
import sys
import traceback
import types
from multiprocessing.connection import Client
from pathlib import Path


def _import_encoder():
    # Import the adapter through a private package so loading it does not
    # trigger LeRobot or the parent latent_video tree.
    package = types.ModuleType("distilled_adapter_runtime")
    package.__path__ = [
        str(Path(__file__).resolve().parents[1] / "src/lerobot_policy_latent_act/encoders")
    ]
    sys.modules[package.__name__] = package
    return importlib.import_module("distilled_adapter_runtime.distilled")


def main():
    startup = json.loads(sys.stdin.readline())
    connection = Client(tuple(startup["address"]), authkey=bytes.fromhex(startup["key"]))
    encoder = None

    def cleanup():
        if encoder is not None:
            encoder.close()

    atexit.register(cleanup)
    try:
        request = connection.recv()
        if request[0] != "init":
            raise ValueError("Distilled worker must be initialized first")
        options, device = request[1], request[2]
        root = Path(options["distilled_root"]).resolve(strict=True)
        # The child may inherit the ACT process's PYTHONPATH. Match the
        # distillation checkout's own launcher: its modified Diffusers and
        # HeFT sources must win over the ACT environment. Do not put V-JEPA2
        # here; the adapter inserts and validates that root before importing
        # its namespace `src` package.
        sys.path[:0] = [
            str(root / "diffusers/src"),
            str(root / "src"),
            str(root),
        ]
        for variable in (
            "HF_HUB_OFFLINE",
            "HF_DATASETS_OFFLINE",
            "TRANSFORMERS_OFFLINE",
        ):
            os.environ[variable] = "1"
        module = _import_encoder()
        import torch

        torch.set_num_threads(2)
        encoder = module.DistilledEncoder(types.SimpleNamespace(**options), device)
        connection.send(("ok", encoder.health()))
        while True:
            request = connection.recv()
            if request[0] == "close":
                break
            if request[0] == "encode":
                result = encoder.encode([torch.from_numpy(array) for array in request[1]])
                # NumPy payloads avoid PyTorch's shared-memory file descriptor
                # service. Float32 transport covers BF16/FP16 model output.
                payload = [features.cpu().float().numpy() for features in result]
                connection.send(("ok", payload))
            else:
                raise ValueError(f"Unknown worker request: {request[0]}")
    except (EOFError, BrokenPipeError):
        pass
    except Exception:  # noqa: BLE001 -- propagate the complete worker failure
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        cleanup()
        connection.close()


if __name__ == "__main__":
    main()
