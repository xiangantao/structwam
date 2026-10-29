#!/usr/bin/env bash
# Usage: measure_dropout_loss.sh NPU CKPT_DIR CKPT_NAME N_BATCHES OUT_JSON
set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
export PYTHONPATH="/home/ma-user/work/2337888765/workspace/StructWAM:/home/ma-user/work/2337888765/workspace/heft/latent_policy/src:/home/ma-user/work/2337888765/workspace/heft"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 WANDB_MODE=disabled ASCEND_RT_VISIBLE_DEVICES=$1
cd /home/ma-user/work/2337888765/workspace/StructWAM
exec /home/ma-user/work/2337888765/envs/openwam-npu/bin/python /home/ma-user/work/2337888765/workspace/runs/structwam_a123_tests_20261027/measure_dropout_loss.py "${@:2}"
