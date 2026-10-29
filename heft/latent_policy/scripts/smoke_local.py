"""One real-data ACT update per backbone; local weights/data only, no formal training run."""

import argparse
import gc
import json
from pathlib import Path

import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from torch.utils.data import default_collate

from lerobot_policy_latent_act import LatentACTConfig
from lerobot_policy_latent_act.data import (
    episodes_with_tasks,
    official_task_names,
    select_suite_episodes,
)
from lerobot_policy_latent_act.modeling_latent_act import LatentACTPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("wan", "vjepa2", "wan_jepa"), required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    torch.set_num_threads(2)
    torch.manual_seed(42)
    names = official_task_names("libero_spatial")
    meta = LeRobotDatasetMetadata("lerobot/libero", root=root / "datasets/libero")
    selected = select_suite_episodes(episodes_with_tasks(meta), names)
    cameras = ["observation.images.image", "observation.images.image2"]
    delta = {key: [i / meta.fps for i in range(-15, 1)] for key in cameras}
    delta["action"] = [i / meta.fps for i in range(16)]
    dataset = LeRobotDataset(
        "lerobot/libero",
        root=root / "datasets/libero",
        episodes=selected["episodes"][:1],
        delta_timestamps=delta,
        video_backend="pyav",
        return_uint8=True,
    )
    sample = dataset[min(15, len(dataset) - 1)]
    print("decoded", {key: list(sample[key].shape) for key in cameras}, flush=True)
    config = LatentACTConfig(
        encoder=args.encoder,
        device="cuda:0",
        task_names=names,
        extract_batch_size=1,
        wan_model_path=str(root / "models/Wan2.1-T2V-1.3B-Diffusers"),
        wan_runtime_python=str(root / "heft/.venv/bin/python"),
        channel_mask=str(
            root / "heft/reports/scannet_channels/73f4e810824b58cb/global_256/masks.json"
        ),
        vjepa_root=str(root / "vjepa2"),
        vjepa_checkpoint=str(root / "models/vjepa2/vitl.pt"),
        jepa_model_path=str(root / "models/vjepa2/vitl.pt"),
        artifact_dir=str(root / "heft/latent_video/wan_jepa/artifacts"),
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (8,)),
            **{key: PolicyFeature(FeatureType.VISUAL, (3, 256, 256)) for key in cameras},
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
    )
    policy = LatentACTPolicy(config).to("cuda:0")
    pre, _ = make_pre_post_processors(config, dataset_stats=meta.stats)
    batch = default_collate([sample])
    for key in cameras:
        batch[key] = batch[key].float() / 255
    batch = pre(batch)
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=config.optimizer_lr)
    try:
        print("forward", args.encoder, flush=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, metrics = policy(batch)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        grad = policy.model.encoder_img_feat_input_proj.weight.grad
        if grad is None or not torch.isfinite(grad).all():
            raise ValueError("Invalid visual projection gradient")
        optimizer.step()
        torch.cuda.synchronize()
        result = {
            "encoder": args.encoder,
            "loss": loss.item(),
            "metrics": metrics,
            "projection_grad_norm": grad.norm().item(),
            "dataset_fps": meta.fps,
            "episode": selected["episodes"][0],
            "task": sample["task"],
            "allocated_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        path = root / f"heft/latent_policy/.runtime/smoke_{args.encoder}.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        policy.close()
    del policy, optimizer, batch
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
