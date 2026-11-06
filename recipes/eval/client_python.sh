#!/usr/bin/env bash
# LIBERO-plus client interpreter. Loads the optional extra system libraries (LIBERO_SYSDEPS_ENV:
# ImageMagick / Mesa-DRI for stock images) into the client only, and runs single_eval.py through
# clean_prompt_entry.py, which strips LIBERO-plus's perturbation ids from the instruction. Every
# other call (version probes, task enumeration) is plain python.
source "$(dirname "${BASH_SOURCE[0]}")/../paths.sh"
[[ -n "$LIBERO_SYSDEPS_ENV" ]] && source "$LIBERO_SYSDEPS_ENV"
# The scheduler sets these for its clients; set them here too so direct calls (list_prompts.py) work.
export LIBERO_PLUS_PATH="${LIBERO_PLUS_PATH:-$LIBERO_PLUS_REPO}"
export PYTHONPATH="$LIBERO_PLUS_REPO:$PYTHONPATH"
if [[ "${1:-}" == */single_eval.py ]]; then
    exec "$LIBERO_PY" "$STRUCTWAM_REPO/recipes/eval/clean_prompt_entry.py" "$@"
fi
exec "$LIBERO_PY" "$@"
