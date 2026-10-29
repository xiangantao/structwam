#!/usr/bin/env bash
# Gradient attribution on the StructWAM s40000 checkpoint. Usage: run.sh NPROC BATCHES BATCH_SIZE OUT [extra args]
set -euo pipefail
R=/home/ma-user/work/2337888765
W=$R/workspace
D=$W/runs/structwam_grad_attr_20261027
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
export PYTHONPATH="$W/StructWAM:$W/heft/latent_policy/src:$W/heft"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 WANDB_MODE=disabled
cd $W/StructWAM
NPROC=$1 BATCHES=$2 BS=$3 OUT=$4; shift 4
exec $R/envs/openwam-npu/bin/torchrun --nnodes=1 --nproc_per_node=$NPROC --master_port=29711 $D/grad_attribution.py \
    --checkpoint $W/runs/structwam_step40000_eval_ckpt/checkpoint_step_40000.safetensors \
    --output $D/$OUT --batches $BATCHES --batch-size $BS "$@"
