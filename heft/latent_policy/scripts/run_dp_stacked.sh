#!/usr/bin/env bash
# Grasp-precision fixes stacked, libero_spatial, 8 NPUs, global batch 64, cosine over 30k steps stopped at
# 10k (as the 10-18 runs), then closed-loop evaluation of both policies with grasp instrumentation:
#  1) vanilla DP with the ImageNet-pretrained ResNet-18 (BatchNorm)          -> fair baseline   (~40 min)
#  2) DP + maps / tracks / trunk with: the same pretrained ResNet; SCDP-style multi-scale image features at
#     the student's keypoints (--anchors); GAP-style phase-guided gradient scaling of the student branches
#     near the demos' gripper open/close events (--phase-strength 1, sigma 8 steps)          (~10-11 h)
#  3) launch_libero_eval.sh --instrument on both, 10 tasks x 50 init states                    (~1.5 h)
# Each training step is skipped when its 10k checkpoint already exists (safe to rerun).
set -uo pipefail
W=/home/ma-user/work/2337888765/workspace
P=$W/heft/latent_policy
R=$W/runs
TRAIN=$R/dp_stacked_20261018
WEIGHTS=${WEIGHTS:-$W/pretrained/resnet18-f37072fd.pth}
PY=/home/ma-user/anaconda3/envs/vace-npu/bin/python
until [ -f "$WEIGHTS" ] && [ "$(sha256sum "$WEIGHTS" | cut -c1-8)" = f37072fd ]; do sleep 60; done
while ps -eo args | grep -q "[e]val_libero_shard.sh"; do sleep 60; done
cd "$W/heft"
source "$P/scripts/env.sh" >/dev/null 2>&1
export PYTHONPATH="$PWD/diffusers/src:$PYTHONPATH" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 VACE_NPU=1 HCCL_CONNECT_TIMEOUT=1200
common=(--suite libero_spatial --manifest "$R/vace_latent_data_npu/episodes.jsonl" --dataset-root "$W/datasets/libero"
        --heldout "$R/vace_teacher_cache_small_20261015/windows_summary.json" --steps 30000 --stop-step 10000
        --val-every 2000 --save-every 5000 --log-every 50 --rgb-weights "$WEIGHTS")
mkdir -p "$TRAIN/rgb" "$TRAIN/maps"
if [ ! -f "$TRAIN/rgb/step_0010000.pt" ]; then
  echo "vanilla DP (pretrained ResNet) start $(date)"
  "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 "$P/scripts/train_dp_rgb.py" "${common[@]}" \
    --output "$TRAIN/rgb" > "$TRAIN/rgb/train.log" 2>&1 < /dev/null
  echo "vanilla DP (pretrained ResNet) exit $? $(date)"
fi
if [ ! -f "$TRAIN/maps/step_0010000.pt" ]; then
  echo "DP+A+C+J stacked start $(date)"
  "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 "$P/scripts/train_dp_maps_online.py" "${common[@]}" \
    --checkpoint "$R/mainline_20261016/g2_full/step_0002000.pt" --roles "$R/policy_maps_20261018/roles.json" \
    --anchors --phase-strength 1.0 --phase-sigma 8 --output "$TRAIN/maps" > "$TRAIN/maps/train.log" 2>&1 < /dev/null
  echo "DP+A+C+J stacked exit $? $(date)"
fi
[ -f "$TRAIN/rgb/step_0010000.pt" ] && [ -f "$TRAIN/maps/step_0010000.pt" ] || { echo "missing checkpoint, no evaluation"; exit 1; }
echo "closed-loop evaluation start $(date)"
OUT=$R/libero_eval_20261018/stacked MAPS_CKPT="$TRAIN/maps/step_0010000.pt" RGB_CKPT="$TRAIN/rgb/step_0010000.pt" \
  EXTRA="--instrument" SHARDS_PER_NPU=4 LP_NUM_THREADS=3 bash "$P/scripts/launch_libero_eval.sh"
