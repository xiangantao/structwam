"""Turn a StructWAM training checkpoint directory into an evaluation checkpoint directory.

    $OPENWAM_PY recipes/prepare_eval_ckpt.py SRC_DIR DST_DIR [--ckpt-name checkpoint_step_40000.safetensors]

SRC_DIR is a training checkpoint dir (config.yaml, normalization_stats.npy, reason1/, the weights),
e.g. the released $STRUCTWAM_ASSETS/checkpoints/b2. DST_DIR gets:
  - the weights, normalization stats and reason1/ as symlinks (nothing is copied),
  - config.yaml with the inference switched to the best setting (1 video denoising step far and
    near, video / structure K/V cache on), and every machine path (student, appearance stats,
    latent_policy, Cosmos models, dataset) pointed at this machine's STRUCTWAM_ASSETS / STRUCTWAM_REPO.
"""

import argparse
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--ckpt-name", default="checkpoint_step_40000.safetensors")
    parser.add_argument("--video-steps", type=int, default=1, help="video denoising steps, far and near (default 1)")
    parser.add_argument("--no-kv-cache", action="store_true", help="keep the video / structure K/V cache off")
    args = parser.parse_args()

    assets = Path(os.environ["STRUCTWAM_ASSETS"])
    repo = Path(os.environ["STRUCTWAM_REPO"])
    src, dst = args.src.resolve(), args.dst
    for name in (args.ckpt_name, "config.yaml", "normalization_stats.npy", "reason1"):
        if not (src / name).exists():
            print(f"missing {src / name}", file=sys.stderr)
            return 1

    cfg = OmegaConf.load(src / "config.yaml")
    model = cfg.model
    structure = model.architecture.structure
    structure.inference.video_steps_far = args.video_steps
    structure.inference.video_steps_near = args.video_steps
    structure.inference.video_kv_cache = not args.no_kv_cache

    student = structure.student
    student.latent_policy_root = str(repo / "heft" / "latent_policy")
    student.checkpoint = str(assets / "student" / "step_0002000.pt")
    student.appearance_stats = str(assets / "student" / "appearance_stats.npz")
    student.text_root = str(assets / "models" / "Wan2.1-T2V-1.3B-Diffusers")

    vb = model.video_backbone
    if "model_path" in vb:
        vb.model_path = str(assets / "models" / "Cosmos-Predict2.5-2B")
    if "text_encoder_path" in vb:
        vb.text_encoder_path = str(assets / "models" / "Cosmos-Reason1-7B")
    if "dataloader" in cfg and "dataset_dir" in cfg.dataloader:
        cfg.dataloader.dataset_dir = str(assets / "datasets" / "openwam" / "libero")

    dst.mkdir(parents=True, exist_ok=True)
    for name in (args.ckpt_name, "normalization_stats.npy", "reason1"):
        link = dst / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(src / name)
    OmegaConf.save(cfg, dst / "config.yaml")
    print(f"eval checkpoint: {dst} (video steps {args.video_steps}, K/V cache {not args.no_kv_cache})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
