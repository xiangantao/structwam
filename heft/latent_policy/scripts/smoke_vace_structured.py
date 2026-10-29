#!/usr/bin/env python3
"""One real window/teacher/backward check; no sustained training is launched."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent))

from vace_latent_distill.config import NEGATIVE_PROMPT, TASKS, PipelineConfig
from vace_latent_distill.data import StructuredDataset
from vace_latent_distill.losses import structured_loss
from vace_latent_distill.model import StructuredStudent
from vace_latent_distill.targets import stack_targets
from vace_latent_distill.teacher import VaceTeacher, content_token_masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/vace_structured.json")
    parser.add_argument("--task", choices=TASKS, default="shared")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backward", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    def progress(stage):
        print(json.dumps({"stage": stage}), flush=True)

    torch.set_num_threads(4)
    torch.manual_seed(42)
    config = PipelineConfig.load(args.config)
    config.model.task = args.task
    progress("read_window")
    dataset = StructuredDataset(
        config.train.manifest,
        config.train.text_cache,
        image_size=config.model.image_size,
        seed=config.train.seed,
        windows_per_episode_camera=1,
    )
    item = dataset[args.index]
    device = torch.device("cuda:0")
    started = time.monotonic()
    progress("load_teacher")
    teacher = VaceTeacher(config.teacher, args.task, device, dataset.text[NEGATIVE_PROMPT])
    content = content_token_masks(config.teacher.text_root, [item["metadata"]["task"]])
    progress("extract_teacher")
    target = teacher.extract(
        item["teacher_rgb"],
        item["text"],
        item["text_mask"],
        item["metadata"]["teacher_seed"],
        content[item["metadata"]["task"]],
    )
    protocol = teacher.protocol.copy()
    teacher.close()
    del teacher
    torch.cuda.empty_cache()
    progress("load_student")
    model = StructuredStudent(config.model).to(device)
    model.train(args.backward)
    torch.cuda.reset_peak_memory_stats()
    progress("student_forward")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = model(
            item["rgb"][None].to(device),
            item["text"][None].to(device),
            item["text_mask"][None].to(device),
        )
        loss, details = structured_loss(
            prediction,
            stack_targets([target], device),
            task=args.task,
            samples=config.train.relation_queries,
            head_weights=config.train.head_weights,
        )
    if args.backward:
        progress("student_backward")
        loss.backward()
        gradients = {
            "backbone": float(model.backbone.patch_embed.proj.weight.grad.float().norm()),
            "future_query": float(model.future_query.grad.float().norm()),
            "text_projection": float(model.text_projection[1].weight.grad.float().norm()),
        }
    else:
        gradients = {}
    result = {
        "task": args.task,
        "sample": item["metadata"],
        "teacher_protocol": protocol,
        "shapes": {
            branch: {key: list(value.shape) for key, value in prediction[branch].items()}
            for branch in ("observed", "future")
        },
        "loss": {key: float(value) for key, value in details.items()},
        "gradients": gradients,
        "peak_student_gib": torch.cuda.max_memory_allocated() / 2**30,
        "elapsed_seconds": time.monotonic() - started,
        "student_future_rgb": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{args.task}.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
