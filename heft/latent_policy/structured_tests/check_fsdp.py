"""Two-rank bounded FSDP2 forward/backward/optimizer/save/restore check."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from test_pipeline import inputs, make_model, teacher_target, tiny_config

from vace_latent_distill.config import TASKS, PipelineConfig, TrainConfig
from vace_latent_distill.losses import structured_loss
from vace_latent_distill.training import (
    finite_gradient_norm,
    load_checkpoint,
    optimizer_for,
    save_checkpoint,
    wrap_student,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, default="shared")
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    dist.init_process_group("nccl", device_id=device)
    config = PipelineConfig(
        model=tiny_config(args.task, checkpointing=True), train=TrainConfig(steps=2)
    )
    model = wrap_student(make_model(args.task, checkpointing=True).to(device), "fsdp2", device)
    optimizer = optimizer_for(model, config.train)
    batch = tuple(value.to(device) for value in inputs())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = model(*batch)
        target = teacher_target(prediction, args.task)
        loss, _ = structured_loss(prediction, target, task=args.task, samples=4)
    loss.backward()
    norm = finite_gradient_norm(model, 1.0)
    optimizer.step()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = model(*batch)["future"]
    provenance = {"synthetic": True, "world_size": dist.get_world_size()}
    checkpoint = args.output / "fsdp_check.pt"
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        config=config,
        step=1,
        epoch=0,
        cursor=1,
        provenance=provenance,
    )
    dist.barrier()
    model2 = wrap_student(make_model(args.task, checkpointing=True).to(device), "fsdp2", device)
    optimizer2 = optimizer_for(model2, config.train)
    assert load_checkpoint(checkpoint, model2, optimizer2, config, provenance) == (1, 0, 1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        actual = model2(*batch)["future"]
    error = max((actual[key] - value).abs().max().item() for key, value in expected.items())
    assert error == 0, error
    if rank == 0:
        result = {
            "ranks": dist.get_world_size(),
            "task": args.task,
            "loss": loss.item(),
            "grad_norm": norm,
            "restore_max_error": error,
            "synthetic_small_model": True,
        }
        (args.output / "fsdp_check.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
