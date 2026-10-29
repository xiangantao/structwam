#!/usr/bin/env bash
# ImageNet-pretrained ResNet-18 (BatchNorm, LeRobot 0.6 default) for both policies on libero_spatial;
# everything else as the 10-18 runs (8 NPUs, global batch 64, cosine over 30k steps stopped at 10k,
# DP+A+C+J features online).  Waits for the torchvision weights (sha256 prefix f37072fd) and for any
# running LIBERO evaluation to finish, then trains vanilla DP (~40 min) and DP+A+C+J (~10 h).
set -uo pipefail
W=/home/ma-user/work/2337888765/workspace
P=$W/heft/latent_policy
R=$W/runs
OUT=$R/dp_pretrained_20261018
WEIGHTS=${WEIGHTS:-$HOME/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth}
PY=/home/ma-user/anaconda3/envs/vace-npu/bin/python
until [ -f "$WEIGHTS" ] && [ "$(sha256sum "$WEIGHTS" | cut -c1-8)" = f37072fd ]; do sleep 60; done
while ps -eo args | grep -q "[e]val_libero_shard.sh"; do sleep 60; done
cd "$W/heft"
source "$P/scripts/env.sh" >/dev/null 2>&1
export PYTHONPATH="$PWD/diffusers/src:$PYTHONPATH" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 VACE_NPU=1 HCCL_CONNECT_TIMEOUT=1200
common=(--suite libero_spatial --manifest "$R/vace_latent_data_npu/episodes.jsonl" --dataset-root "$W/datasets/libero"
        --heldout "$R/vace_teacher_cache_small_20261015/windows_summary.json" --steps 30000 --stop-step 10000
        --val-every 2000 --save-every 5000 --rgb-weights "$WEIGHTS")
mkdir -p "$OUT/rgb" "$OUT/maps"
echo "vanilla DP (pretrained) start $(date)"
"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 "$P/scripts/train_dp_rgb.py" "${common[@]}" \
  --log-every 50 --output "$OUT/rgb" > "$OUT/rgb/train.log" 2>&1 < /dev/null
echo "vanilla DP (pretrained) exit $? $(date)"
"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 "$P/scripts/train_dp_maps_online.py" "${common[@]}" \
  --log-every 50 --checkpoint "$R/mainline_20261016/g2_full/step_0002000.pt" --roles "$R/policy_maps_20261018/roles.json" \
  --output "$OUT/maps" > "$OUT/maps/train.log" 2>&1 < /dev/null
echo "DP+A+C+J (pretrained) exit $? $(date)"
