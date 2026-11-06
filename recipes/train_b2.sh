#!/usr/bin/env bash
# Train StructWAM with the b2 recipe (StructWAM/configs/train_struct_b2.yaml) on one node of 8 Ascend NPUs.
#
#   bash recipes/train_b2.sh RUN_DIR [hydra overrides...]
#   bash recipes/train_b2.sh --dry-run RUN_DIR     # print the resolved config only
#
# 40k micro-steps, batch 8 per NPU x accumulation 2 (effective 128), checkpoint every 2000 steps,
# keeps the last 2. Re-running with the same RUN_DIR resumes from its latest saved state.
# STOP_STEP=N ends the job at step N with a resumable state (the LR schedule still follows 40k).
# NPROC (default 8) / MASTER_PORT (default 29623) can be overridden.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

dry=0
[[ "${1:-}" == "--dry-run" ]] && { dry=1; shift; }
RUN_DIR="${1:?usage: train_b2.sh [--dry-run] RUN_DIR [overrides...]}"
shift
mkdir -p "$RUN_DIR"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"

if [[ -f "$ASCEND_SET_ENV" ]]; then
    set +eu +o pipefail
    source "$ASCEND_SET_ENV"
    set -euo pipefail
fi
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 WANDB_MODE="${WANDB_MODE:-disabled}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
PYBIN="$(dirname "$OPENWAM_PY")"
export PATH="$PYBIN:$PATH"

args=(--config-name=train_struct_b2 "training.output_path=$RUN_DIR/checkpoints" "project.output_dir=$RUN_DIR/hydra")
[[ -n "${STOP_STEP:-}" ]] && args+=("training.stop_step=$STOP_STEP")
latest=$(ls -d "$RUN_DIR"/checkpoints/*/accel_state_step_* 2>/dev/null | sort -V | tail -1 || true)
[[ -n "$latest" ]] && { args+=("training.resume_ckpt_path=$(dirname "$latest")"); echo "Resuming from $latest"; }

cd "$STRUCTWAM_REPO/StructWAM"
if (( dry )); then
    exec "$OPENWAM_PY" scripts/train.py "${args[@]}" "$@" --cfg job --resolve
fi
[[ -f "$RUN_DIR/train.log" ]] && mv "$RUN_DIR/train.log" "$RUN_DIR/train.log.$(date +%Y%m%d_%H%M%S)"
{ echo "date=$(date) host=$(hostname)"; git -C "$STRUCTWAM_REPO" rev-parse HEAD; git -C "$STRUCTWAM_REPO" status --short; } \
    > "$RUN_DIR/provenance.txt"
cmd=("$PYBIN/torchrun" --nnodes=1 --nproc_per_node="${NPROC:-8}" "--master_port=${MASTER_PORT:-29623}"
     scripts/structwam/npu_entry.py scripts/train.py "${args[@]}" "$@")
printf '%q ' "${cmd[@]}" > "$RUN_DIR/command.txt"; echo >> "$RUN_DIR/command.txt"
"${cmd[@]}" 2>&1 | tee "$RUN_DIR/train.log"
