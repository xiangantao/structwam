#!/usr/bin/env bash
# Run on BOTH instances with the same MASTER_ADDR and RUN_NAME.
# Node 0: NODE_RANK=0 MASTER_ADDR=10.x.x.x RUN_NAME=libero_16npu_run01 bash scripts/structwam/train_libero_16npu.sh
# Node 1: NODE_RANK=1 MASTER_ADDR=10.x.x.x RUN_NAME=libero_16npu_run01 bash scripts/structwam/train_libero_16npu.sh
# MASTER_ADDR must be node 0's reachable private IP, not its SSH gateway.
# Both nodes need identical code/dependencies and inputs at the configured paths.
# Use shared RUN_ROOT storage for complete distributed checkpoints/resume.
# Optional per-node network selection: HCCL_SOCKET_IFNAME / GLOO_SOCKET_IFNAME.
# Additional Hydra overrides may be appended, including training.resume_ckpt_path.
set -euo pipefail
: "${NODE_RANK:?Set NODE_RANK=0 on the first instance, 1 on the second}"
: "${MASTER_ADDR:?Set MASTER_ADDR to node 0 private IP on both instances}"
: "${RUN_NAME:?Set the same unique RUN_NAME on both instances}"
[[ "$NODE_RANK" == 0 || "$NODE_RANK" == 1 ]] || { echo "NODE_RANK must be 0 or 1" >&2; exit 2; }
[[ "$RUN_NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "RUN_NAME: use letters, numbers, underscore or hyphen" >&2; exit 2; }
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE=/home/ma-user/work/2337888765/workspace
ENV_ROOT="${ENV_ROOT:-/home/ma-user/work/2337888765/envs/openwam-npu}"
RUN_ROOT="${RUN_ROOT:-$WORKSPACE/runs}"
RUN_DIR="$RUN_ROOT/$RUN_NAME"
export PATH="$ENV_ROOT/bin:$PATH"
export PYTHONPATH="$REPO_ROOT:$WORKSPACE/heft/latent_policy/src:$WORKSPACE/heft${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
cd "$REPO_ROOT"
[[ -x "$ENV_ROOT/bin/torchrun" ]] || { echo "Missing torchrun in $ENV_ROOT" >&2; exit 1; }
cmd=("$ENV_ROOT/bin/torchrun" --nnodes=2 --nproc_per_node=8
    "--node_rank=$NODE_RANK" "--master_addr=$MASTER_ADDR"
    "--master_port=${MASTER_PORT:-29623}" --max_restarts=0
    scripts/structwam/npu_entry.py scripts/train.py
    --config-name=train_struct_16npu
    "training.output_path=$RUN_DIR/checkpoints"
    "project.output_dir=$RUN_DIR/hydra/node$NODE_RANK"
    "$@")
mkdir -p "$RUN_DIR"
# Refuse accidental reuse of this node's logs; use a new RUN_NAME for each launch.
LOG="$RUN_DIR/node$NODE_RANK.log"
if ! (set -o noclobber; : > "$LOG") 2>/dev/null; then
    echo "Log already exists: $LOG. Choose a new RUN_NAME." >&2
    exit 1
fi
printf '%q ' "${cmd[@]}" > "$RUN_DIR/command_node$NODE_RANK.txt"
printf '\n' >> "$RUN_DIR/command_node$NODE_RANK.txt"
echo "Node $NODE_RANK / 2; 8 NPUs per node; master $MASTER_ADDR:${MASTER_PORT:-29623}"
echo "20000 micro-steps; batch 8 per NPU; accumulation 1; effective batch 128"
echo "Log: $LOG"
"${cmd[@]}" 2>&1 | tee -a "$LOG"
