#!/usr/bin/env bash
# Score emergent segmentation on rendered scenes. Usage: score.sh SCENES OUT [NPU]
set -euo pipefail
R=/home/ma-user/work/2337888765
W=$R/workspace
D=$W/runs/student_seg_robustness_20261027
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
export PYTHONPATH="$W/heft/latent_policy/src:$W/heft" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
export ASCEND_RT_VISIBLE_DEVICES=${3:-0}
cd $W/heft/latent_policy
exec $R/envs/openwam-npu/bin/python $D/score_seg.py --scenes $D/$1 --output $D/$2
