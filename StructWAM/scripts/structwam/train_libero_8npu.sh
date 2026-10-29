#!/usr/bin/env bash
# Full run: bash scripts/structwam/train_libero_8npu.sh
# Inspect without training: bash scripts/structwam/train_libero_8npu.sh --dry-run
# Resume: bash scripts/structwam/train_libero_8npu.sh training.resume_ckpt_path=/path/to/checkpoint/run
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_ROOT=/home/ma-user/work/2337888765/envs/openwam-npu
WORKSPACE=/home/ma-user/work/2337888765/workspace
export PATH="$ENV_ROOT/bin:$PATH"
export PYTHONPATH="$REPO_ROOT:$WORKSPACE/heft/latent_policy/src:$WORKSPACE/heft${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=1
    shift
fi
RUN_DIR="${RUN_DIR:-$WORKSPACE/runs/structwam_libero_40k_$(date +%Y%m%d_%H%M%S)_$$}"
cd "$REPO_ROOT"
[[ -x "$ENV_ROOT/bin/torchrun" ]] || { echo "Missing torchrun: $ENV_ROOT/bin/torchrun" >&2; exit 1; }
for required in "$WORKSPACE/datasets/openwam/libero/meta/conversion.json" \
    "$WORKSPACE/models/Cosmos-Predict2.5-2B" "$WORKSPACE/models/Cosmos-Reason1-7B" \
    "$WORKSPACE/runs/mainline_20261016/g2_full/step_0002000.pt" \
    "$WORKSPACE/runs/dp_stacked_20261018/maps/stats.npz"; do
    [[ -e "$required" ]] || { echo "Missing input: $required" >&2; exit 1; }
done
cmd=("$ENV_ROOT/bin/torchrun" --nnodes=1 --nproc_per_node=8
    "--master_port=${MASTER_PORT:-29623}"
    scripts/structwam/npu_entry.py scripts/train.py --config-name=train_struct
    "training.output_path=$RUN_DIR/checkpoints"
    "project.output_dir=$RUN_DIR/hydra"
    "$@")
if (( dry_run )); then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    exec "$ENV_ROOT/bin/python" scripts/train.py --config-name=train_struct --cfg job --resolve "training.output_path=$RUN_DIR/checkpoints" "project.output_dir=$RUN_DIR/hydra" "$@"
fi
mkdir -p "$RUN_DIR"
printf '%q ' "${cmd[@]}" > "$RUN_DIR/command.txt"
printf '\n' >> "$RUN_DIR/command.txt"
echo "Run directory: $RUN_DIR"
echo "Log: $RUN_DIR/train.log"
"${cmd[@]}" 2>&1 | tee "$RUN_DIR/train.log"
