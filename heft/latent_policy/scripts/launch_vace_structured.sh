#!/usr/bin/env bash
set -euo pipefail

LATENT_STRUCTURED_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$LATENT_STRUCTURED_DIR/env.sh"
cd "$LATENT_HEFT_ROOT"

latent_structured_task="${1:-shared}"
case "$latent_structured_task" in
  shared|seg_l22h9|seg_l15h7|tracking|semantic|depth) ;;
  *) printf 'Unknown task: %s\n' "$latent_structured_task" >&2; exit 2 ;;
esac
if [[ $# -gt 0 ]]; then shift; fi

exec "$LATENT_POLICY_PYTHON" -m torch.distributed.run --standalone \
  --nproc_per_node="${NPROC_PER_NODE:-8}" \
  "$LATENT_STRUCTURED_DIR/train_vace_structured.py" \
  --config "$LATENT_POLICY_ROOT/configs/vace_structured.json" \
  --task "$latent_structured_task" "$@"
