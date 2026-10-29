#!/usr/bin/env bash
set -euo pipefail

# Resume an already prepared data run on another machine/GPU allocation.
# This changes only worker placement; it does not rebuild windows or labels.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
cd "$LATENT_HEFT_ROOT"

run="${VACE_CACHE_RUN:-$LATENT_WORKSPACE/runs/vace_teacher_offline_libero8_20261009}"
gpus="${VACE_CACHE_GPUS:-0,1,2,3,4,5,6,7}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) run="$2"; shift 2 ;;
    --gpus) gpus="$2"; shift 2 ;;
    --help|-h)
      printf 'Usage: %s [--run PATH] [--gpus 0,1,...,7]\n' "$0"
      exit 0
      ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done

exec "$LATENT_POLICY_PYTHON" "$SCRIPT_DIR/cache_vace_teacher.py" \
  --role start --run "$run" --override-gpus "$gpus"
