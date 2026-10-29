#!/usr/bin/env bash
# StructWAM training on the ModelArts notebook (8x Ascend 910B3).
#   bash scripts/structwam/train_npu.sh dataloader.dataset_dir=/path/to/libero [overrides]
# Not run yet: needs the Cosmos-Predict2.5 / Cosmos-Reason1 weights, the cosmos_predict25 extra,
# OpenWAM-format LIBERO data and the NPU port checks listed in STRUCTWAM.md.
set -euo pipefail
cd "$(dirname "$0")/../.."
W=/home/ma-user/work/2337888765/workspace
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
# The torch-only transformer_engine stand-in (npu_shims/) is put on sys.path by the Cosmos builder
# (cosmos_predict25/npu_compat.py) after megatron-core has loaded; do not add it to PYTHONPATH.
export PYTHONPATH="$PWD:$W/heft/latent_policy/src:$W/heft:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
exec torchrun --nnodes 1 --nproc_per_node "$NPROC_PER_NODE" --master_port "${MASTER_PORT:-29500}" \
    scripts/structwam/npu_entry.py scripts/train.py --config-name train_struct "$@"
