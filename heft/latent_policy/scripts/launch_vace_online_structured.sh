#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
cd "$LATENT_HEFT_ROOT"

# Online teacher extraction is deliberately independent of the offline/TOS cache.
export VACE_CFG_PARALLEL="${VACE_CFG_PARALLEL:-1}"

task="${1:-shared}"
case "$task" in
  shared|seg_l22h9|seg_l15h7|tracking|semantic|depth) ;;
  *) printf 'Unknown task: %s\n' "$task" >&2; exit 2 ;;
esac
if [[ $# -gt 0 ]]; then shift; fi

exec "$LATENT_POLICY_PYTHON" -m torch.distributed.run --standalone \
  --nproc_per_node="${NPROC_PER_NODE:-8}" \
  "$SCRIPT_DIR/train_vace_structured.py" \
  --config "$LATENT_POLICY_ROOT/configs/vace_structured.json" \
  --task "$task" \
  --batch-size "${VACE_STUDENT_BATCH_SIZE:-32}" \
  --teacher-batch-size "${VACE_TEACHER_BATCH_SIZE:-8}" \
  "$@"
