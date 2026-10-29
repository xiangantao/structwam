#!/usr/bin/env bash
set -euo pipefail

latent_cache_scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$latent_cache_scripts/env.sh"
cd "$LATENT_HEFT_ROOT"

"$LATENT_POLICY_PYTHON" "$latent_cache_scripts/cache_vace_teacher.py" --role prepare "$@"
exec "$LATENT_POLICY_PYTHON" "$latent_cache_scripts/cache_vace_teacher.py" --role start "$@"
