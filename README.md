# StructWAM workspace

Code for StructWAM: an OpenWAM fork that adds a frozen VACE-distilled structure student
(V-JEPA2 based) as a third expert next to the Cosmos-Predict2.5-2B video DiT and the action DiT.
The layout mirrors the ModelArts workspace `/home/ma-user/work/2337888765/workspace`
(a symlink to `/mnt/sfs-turbo-wangfei/2337888765/workspace`).

| Path | Contents |
|---|---|
| `StructWAM/` | The model code. History of OpenWAM-Official/OpenWAM up to upstream `4d7f642` (here `7c3cd8d`: same trees and authors, co-author trailers dropped), then the StructWAM commits |
| `heft/latent_policy`, `heft/latent_video` | The structure student and its distillation code (snapshot of xiangantao/good_state at `2fd6a9d` plus uncommitted work) |
| `runs/` | Training-pool jobs, LIBERO / LIBERO-plus evaluation wrappers, tests and diagnostics, the `command.txt` of each run |
| `sysdeps/hce2_aarch64/env.sh` | Environment for the ImageMagick / Mesa-DRI libraries LIBERO-plus clients need on stock pool images (the libraries themselves are not in the repo) |

Not in the repo: checkpoints, datasets, pretrained weights, conda envs (`envs/openwam-npu`,
`envs/libero-plus`), evaluation outputs and the sysdeps binaries; they stay on the SFS.
V-JEPA2 is facebookresearch/vjepa2 at `204698b`, unmodified.

## Restoring on a new machine

The scripts use absolute paths under `/home/ma-user/work/2337888765` and
`/mnt/sfs-turbo-wangfei/2337888765`. Either clone this repo as that `workspace/` directory, or
replace those prefixes. `tri_system_struct.yaml` also names `latent_policy_root`.

```bash
export PYTHONPATH="$WORKSPACE/StructWAM:$WORKSPACE/heft/latent_policy/src:$WORKSPACE/heft"
```

## Entry points

- Training arm (8 NPUs, auto-resume): `runs/structwam_a123_20261027/train_arm_modelarts.sh {a1|a2|a3|b2|b0|c0}`
- LIBERO-plus 10% clean protocol: `runs/libero_plus_sample10_clean/eval_libero_plus_modelarts.sh NAME CKPT_DIR CKPT_NAME`
- OpenWAM-Cosmos baseline evaluation: `runs/stage0_libero_plus_20261027/eval_openwam_baseline_modelarts.sh`
- What StructWAM adds to OpenWAM: `StructWAM/STRUCTWAM.md`
