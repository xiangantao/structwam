"""One real LIBERO observation, one ACT update, and checkpoint roundtrip; no run launch."""

import argparse
import json
import time
from pathlib import Path

import draccus
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from safetensors.torch import load_model, save_model
from torch.utils.data import default_collate

import lerobot_policy_latent_act  # noqa: F401
from lerobot_policy_latent_act.modeling_latent_act import LatentACTPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new check output directory")
    raw = json.loads(args.config.read_text())
    cfg = draccus.decode(PreTrainedConfig, raw["policy"])
    if cfg.encoder != "distilled" or cfg.dim_model != 256:
        parser.error("Expected distilled fused-only ACT256 config")
    cfg.device = args.device
    cfg.extract_batch_size = 1
    torch.set_num_threads(2)
    torch.manual_seed(42)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    # LeRobot owns timestamps, boundary padding, action chunks and video decoding.
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    data = raw["dataset"]
    meta = LeRobotDatasetMetadata(data["repo_id"], root=Path(data["root"]))
    delta = {key: [i / meta.fps for i in range(-15, 1)] for key in cfg.camera_keys}
    delta["action"] = [i / meta.fps for i in range(cfg.chunk_size)]
    dataset = LeRobotDataset(
        data["repo_id"], root=Path(data["root"]), episodes=[data["episodes"][0]],
        delta_timestamps=delta, video_backend="pyav", return_uint8=True,
    )
    sample = dataset[min(15, len(dataset) - 1)]
    print("decoded real LIBERO history", sample["task"], flush=True)
    batch = default_collate([sample])
    for key in cfg.camera_keys:
        batch[key] = batch[key].float() / 255
    pre, _ = make_pre_post_processors(cfg, dataset_stats=meta.stats)
    batch = pre(batch)
    policy = LatentACTPolicy(cfg).to(device)
    args.output.mkdir(parents=True)
    try:
        policy.train()
        optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=cfg.optimizer_lr)
        projection = policy.model.encoder_img_feat_input_proj
        before = projection.weight.detach().clone()
        started = time.monotonic()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, metrics = policy(batch)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite ACT loss")
        loss.backward()
        grad = projection.weight.grad
        if grad is None or not torch.isfinite(grad).all() or grad.norm() == 0:
            raise RuntimeError("Missing/nonfinite/zero adapter gradient")
        optimizer.step()
        if torch.equal(before, projection.weight):
            raise RuntimeError("Adapter did not update")
        if any("runtime" in k or "backbone" in k for k in policy.state_dict()):
            raise RuntimeError("Frozen student leaked into ACT checkpoint")
        elapsed = time.monotonic() - started
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            expected = policy.predict_action_chunk(batch)
        checkpoint = args.output / "check_policy.safetensors"
        save_model(policy, checkpoint)
        restored = LatentACTPolicy(cfg).to(device)
        try:
            load_model(restored, checkpoint, strict=True)
            # Share the already-loaded immutable runtime to avoid duplicating student memory.
            restored.model._visual_runtime = policy.model._visual_runtime
            restored.model._runtime_device = device
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                actual = restored.predict_action_chunk(batch)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            restored.model._visual_runtime = None
            restored.close()
        report = {
            "config": str(args.config.resolve()), "checkpoint": cfg.distilled_checkpoint,
            "checkpoint_sha256": cfg.distilled_checkpoint_sha256,
            "suite": cfg.suite, "episode": data["episodes"][0], "task": sample["task"],
            "input_shapes": {key: list(batch[key].shape) for key in cfg.camera_keys},
            "projection_shape": list(projection.weight.shape),
            "action_shape": list(actual.shape), "loss": loss.item(), "metrics": metrics,
            "projection_grad_norm": grad.norm().item(), "adapter_updated": True,
            "checkpoint_roundtrip_exact": True, "student_in_policy_state_dict": False,
            "update_seconds_including_student_startup": elapsed,
            "parent_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
            "memory_note": "Parent ACT only; subprocess student memory is excluded. Not a throughput benchmark.",
        }
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        policy.close()


if __name__ == "__main__":
    main()
