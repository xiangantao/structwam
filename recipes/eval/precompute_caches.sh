#!/usr/bin/env bash
# Build the two caches fast-mode evaluation needs, inside the eval checkpoint dir (one NPU, ~5 min):
#   CKPT_DIR/prompts_all.json         every instruction the LIBERO-plus client sends (1549)
#   CKPT_DIR/prompt_cache.pt          Reason1 + crossattn_proj context per prompt (checkpoint-specific)
#   CKPT_DIR/student_text_extra.pt    structure-student UMT5 embeddings for prompts outside its cache (~6 GB)
#
#   bash recipes/eval/precompute_caches.sh CKPT_DIR [CKPT_NAME]      (NPU: DEVICE, default 0)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../paths.sh"
EVAL_DIR="$STRUCTWAM_REPO/recipes/eval"
CKPT_DIR="$(cd "${1:?usage: precompute_caches.sh CKPT_DIR [CKPT_NAME]}" && pwd)"
CKPT_NAME="${2:-checkpoint_step_40000.safetensors}"
if [[ -f "$ASCEND_SET_ENV" ]]; then
    set +eu +o pipefail
    source "$ASCEND_SET_ENV"
    set -euo pipefail
fi
export ASCEND_RT_VISIBLE_DEVICES="${DEVICE:-0}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
NPU_ENTRY="$STRUCTWAM_REPO/StructWAM/scripts/structwam/npu_entry.py"
cd "$STRUCTWAM_REPO/StructWAM"

PROMPTS="$CKPT_DIR/prompts_all.json"
[[ -f "$PROMPTS" ]] || "$EVAL_DIR/client_python.sh" "$EVAL_DIR/list_prompts.py" "$PROMPTS"

if [[ ! -f "$CKPT_DIR/prompt_cache.pt" ]]; then
    "$OPENWAM_PY" "$NPU_ENTRY" "$EVAL_DIR/precompute_prompt_cache.py" \
        "$CKPT_DIR" "$CKPT_NAME" "$PROMPTS" "$CKPT_DIR/prompt_cache.pt.tmp"
    mv "$CKPT_DIR/prompt_cache.pt.tmp" "$CKPT_DIR/prompt_cache.pt"
fi

if [[ ! -f "$CKPT_DIR/student_text_extra.pt" ]]; then
    student=$("$OPENWAM_PY" -c "from omegaconf import OmegaConf; print(OmegaConf.load('$CKPT_DIR/config.yaml').model.architecture.structure.student.checkpoint)")
    "$OPENWAM_PY" "$NPU_ENTRY" "$EVAL_DIR/precompute_student_text.py" \
        "$student" "$STRUCTWAM_ASSETS/models/Wan2.1-T2V-1.3B-Diffusers" "$PROMPTS" "$CKPT_DIR/student_text_extra.pt.tmp"
    mv "$CKPT_DIR/student_text_extra.pt.tmp" "$CKPT_DIR/student_text_extra.pt"
fi
ls -la "$CKPT_DIR/prompt_cache.pt" "$CKPT_DIR/student_text_extra.pt"
