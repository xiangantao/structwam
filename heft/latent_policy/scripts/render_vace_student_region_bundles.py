#!/usr/bin/env python3
"""Render a student checkpoint with the teacher region-bundle readout, on CPU.

The student only receives the past 33 training-domain RGB frames and the cached
instruction embedding.  Its six fields go through the same prompt/K region and
QK/beam decoder as ``render_vace_teacher_region_bundles.py``.  The 16 future
frames are actual recorded RGB used as display background only.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import render_vace_teacher_region_bundles as rb
from vace_latent_distill.bundles import phrase_positions, read_region_bundles
from vace_latent_distill.config import ModelConfig, PipelineConfig
from vace_latent_distill.data import decode_indices, resize_rgb
from vace_latent_distill.model import StructuredStudent
from vace_latent_distill.teacher import content_token_masks

# Same windows and phrases as the teacher demo.  Training-domain videos are the
# LeRobot recordings of the tracking episodes' source episodes (39 -> 1653,
# 40 -> 850); they are horizontally mirrored relative to the tracking renders.
PLANS = [
    {"episode": 1653, "tracking_episode": 39, "start": 48,
     "phrases": ["black bowl", "plate", "wooden cabinet"]},
    {"episode": 850, "tracking_episode": 40, "start": 32,
     "phrases": ["alphabet soup", "basket"]},
]
CAMERAS = {"observation.images.image": "agentview",
           "observation.images.image2": "wrist"}
FIELDS = ("seg_k_l22h9", "seg_k_l15h7", "track_q", "track_k", "text_attention", "depth")


def write_h264(path, frames, fps=10):
    """PyAV/openh264 writer; this host's ffmpeg binary has no libx264."""
    import av

    with av.open(str(path), "w") as container:
        stream = container.add_stream("libopenh264", rate=fps)
        stream.height, stream.width = frames.shape[1:3]
        stream.pix_fmt = "yuv420p"
        stream.bit_rate = 12_000_000
        for frame in frames:
            container.mux(stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")))
        container.mux(stream.encode())


def depth_sheet(case, output, source="student"):
    """Relative depth (fixed-scale readout, not metric) beside RGB."""
    rows, frames = [], (0, 16, 32, 41, 48)
    values = np.concatenate([case["target"][b]["depth"].numpy().ravel() for b in ("observed", "future")])
    low, high = np.percentile(values, (1, 99))
    video = []
    for frame, pixels in enumerate(case["rgb"]):
        rgb = cv2.resize(pixels, (512, 512))
        branch, native = rb.timeline_index(frame)
        depth = rb.at_time(case["target"][branch]["depth"].numpy(), native).reshape(32, 32)
        depth = np.clip((depth - low) / max(high - low, 1e-8), 0, 1)
        depth = cv2.resize(np.uint8(depth * 255), (512, 512), interpolation=cv2.INTER_LINEAR)
        colored = cv2.cvtColor(cv2.applyColorMap(depth, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        blend = np.uint8(rgb * 0.35 + colored * 0.65)
        tag = "1 update" if source == "teacher" else "predicted"
        phase = f"OBS {frame:02d}" if branch == "observed" else f"FUT {frame - 33:02d} | {tag}"
        video.append(np.concatenate((rb.tile(rgb, f"Actual RGB | {phase}"),
                                     rb.tile(blend, f"{source} relative depth")), axis=1))
        if frame in frames:
            rows.append([rgb, blend])
    stem = f"ep{case['episode']} {case['camera']} | {source} relative depth (p1-p99, not metric)"
    rb.labeled_sheet(rows, ("RGB", "Relative depth"), frames, stem).save(output / "depth.jpg", quality=95)
    write_h264(output / "depth.mp4", np.stack(video))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--resume", action="store_true",
                        help="Reuse student_fields.pt already written to --output")
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        parser.error("Use a new output directory")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    rb.write_h264 = write_h264
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(2)
    rb.FONT = str(Path(cv2.__file__).parent / "qt/fonts/DejaVuSans.ttf")
    started = time.monotonic()

    rb.progress(stage="load_checkpoint", checkpoint=str(args.checkpoint))
    saved = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False, mmap=True)
    if saved.get("schema") != "vace-structured-student-v1":
        raise ValueError("Expected a structured student checkpoint")
    config = PipelineConfig.load(args.checkpoint.parent / "pipeline.json")
    if (config.teacher.denoise_updates, config.teacher.denoise_reference_updates) != (1, 13):
        parser.error("This demo follows the one-update mainline")
    model = StructuredStudent(ModelConfig(**saved["config"]["model"]), pretrained=False)
    model.load_state_dict(saved["model"], strict=True)
    step = saved.get("step") or int(args.checkpoint.stem.rsplit("_", 1)[-1])
    del saved
    model.float().eval()

    text_cache = torch.load(config.train.text_cache, map_location="cpu", weights_only=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(Path(config.teacher.text_root) / "tokenizer",
                                              local_files_only=True)
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    cases = []
    for plan in PLANS:
        for camera, label in CAMERAS.items():
            row = next(r for r in rows if r["episode"] == plan["episode"] and r["camera"] == camera)
            absolute = row["episode_start_frame"] + plan["start"]
            video = decode_indices(row["video"], range(absolute, absolute + 49), row["fps"])
            case = {**plan, "camera": label, "prompt": row["task"], "video": row["video"],
                    "frame_indices": [absolute, absolute + 48],
                    "rgb": video.transpose(0, 2, 3, 1).copy(),
                    "student_rgb": resize_rgb(torch.from_numpy(video[:33]), config.model.image_size)}
            case["directory"] = args.output / f"episode_{plan['episode']:05d}_{label}"
            case["directory"].mkdir(exist_ok=args.resume)
            case["phrase_masks"] = phrase_positions(tokenizer, case["prompt"], plan["phrases"])
            cases.append(case)
    content = content_token_masks(config.teacher.text_root, [case["prompt"] for case in cases])

    for case in cases:
        if args.resume and (case["directory"] / "student_fields.pt").is_file():
            case["target"] = torch.load(case["directory"] / "student_fields.pt", map_location="cpu",
                                        weights_only=True)["target"]
            rb.progress(stage="student_fields_reused", episode=case["episode"], camera=case["camera"])
            continue
        text = text_cache[case["prompt"]].float()
        begin = time.monotonic()
        output = model(case["student_rgb"][None], text[None], (text.abs().sum(-1) != 0)[None])
        case["target"] = {branch: {key: output[branch][key][0].float() for key in FIELDS}
                          for branch in ("observed", "future")}
        torch.save({"target": case["target"], "checkpoint": str(args.checkpoint), "step": step},
                   case["directory"] / "student_fields.pt")
        rb.progress(stage="student_forward", episode=case["episode"], camera=case["camera"],
                    seconds=round(time.monotonic() - begin, 1))
    del model

    rb.fit_display(cases)
    summary = []
    for case in cases:
        rb.progress(stage="decode_regions", episode=case["episode"], camera=case["camera"])
        case["decoded"] = {"names": ["Full instruction", *case["phrase_masks"]]}
        for branch in ("observed", "future"):
            case["decoded"][branch] = read_region_bundles(
                case["target"][branch], branch, case["phrase_masks"], content[case["prompt"]],
                config.teacher)
        metrics = rb.render_case(case, case["directory"], source=f"student step{step}")
        depth_sheet(case, case["directory"])
        summary.append({**{key: case[key] for key in ("episode", "tracking_episode", "camera", "prompt",
                                                       "phrases", "start", "frame_indices", "video")},
                        "directory": str(case["directory"]), "diagnostics": metrics})
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        rb.progress(stage="case_rendered", episode=case["episode"], camera=case["camera"])
    (args.output / "protocol.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "step": step, "device": "cpu", "dtype": "float32",
        "student_input": "Past 33 training-domain RGB frames resized to 256 + cached UMT5 instruction",
        "student_future_rgb": False,
        "background": "Actual recorded RGB including 16 future frames, display only",
        "readout": "Same prompt/K region and QK/beam decoder as render_vace_teacher_region_bundles.py",
        "episodes_in_training_set": True,
        "elapsed_seconds": time.monotonic() - started,
    }, indent=2) + "\n")
    rb.progress(stage="completed", output=str(args.output), seconds=time.monotonic() - started)


if __name__ == "__main__":
    main()
