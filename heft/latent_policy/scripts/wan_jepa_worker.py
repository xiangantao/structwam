"""Worker for direct Wan original-five + JEPA supervised-256 extraction."""

import json
import os
import sys
import time
import traceback
from multiprocessing.connection import Client
from pathlib import Path
from types import SimpleNamespace


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
        import torch

        torch.set_num_threads(2)
        from latent_video.wan_jepa.encoder import CombinedExtractor

        while True:
            request = connection.recv()
            if request[0] == "close":
                break
            if request[0] == "init":
                options, device = request[1], request[2]
                encoder = CombinedExtractor.from_local(
                    options["wan_model_path"],
                    channel_mask=options["channel_mask"],
                    device=device,
                    jepa_model_path=options["jepa_model_path"],
                    vjepa_root=Path(options["vjepa_root"]),
                    artifact_dir=options["artifact_dir"],
                )
                connection.send(("ok", "ready"))
            elif request[0] == "encode":
                arrays = request[1]
                frames = [torch.from_numpy(array) for array in arrays]
                batch_size = len(frames[0])
                cameras = len(frames)
                clips = [clip for camera in frames for clip in camera]
                started = time.perf_counter()
                rows = encoder.extract_batch(
                    clips,
                    seeds=[int(options["feature_seed"])] * len(clips),
                    output_device=device,
                )
                outputs = []
                for camera in range(cameras):
                    camera_rows = rows[camera * batch_size : (camera + 1) * batch_size]
                    fused = []
                    for row in camera_rows:
                        wan = row.tensors["wan_stride8"]
                        jepa = row.tensors["jepa_supervised256"]
                        if tuple(wan.shape) != (896, 8, 16, 16) or tuple(jepa.shape) != (256, 8, 16, 16):
                            raise ValueError(f"Unexpected direct feature shapes: {wan.shape}, {jepa.shape}")
                        fused.append(torch.cat((wan, jepa), dim=0))
                    outputs.append(torch.stack(fused).permute(0, 2, 1, 3, 4).float().cpu().numpy())
                connection.send(("ok", outputs))
            else:
                raise ValueError(f"Unknown worker request: {request[0]}")
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        connection.send(("error", traceback.format_exc()))
    finally:
        if encoder is not None:
            encoder.close()
        connection.close()


if __name__ == "__main__":
    main()
