#!/usr/bin/env bash
# Download everything StructWAM needs from Hugging Face into $STRUCTWAM_ASSETS (see recipes/paths.sh).
#
#   bash recipes/download_assets.sh            # everything (~120 GB)
#   bash recipes/download_assets.sh eval       # only what evaluating the released b2 checkpoint needs
#
# Parts: cosmos (video expert, ~29 GB incl. Reason1), umt5 (student text encoder, ~11 GB),
# student (~6.5 GB), b2 (StructWAM b2 40k, ~21 GB), libero (training data, ~2 GB).
# Uses huggingface_hub from the openwam-npu env; set HF_ENDPOINT for a mirror and log in
# (hf auth login) if a repo asks for it. Re-running skips files that are already complete.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

what="${1:-all}"
case "$what" in
    all) parts=(cosmos umt5 student b2 libero) ;;
    eval) parts=(cosmos umt5 student b2) ;;
    *) parts=("$@") ;;
esac

hf_get() {  # repo_type repo_id local_dir [allow_pattern ...]
    "$OPENWAM_PY" - "$@" <<'EOF'
import sys
from huggingface_hub import snapshot_download

repo_type, repo_id, local_dir, *patterns = sys.argv[1:]
snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=local_dir, allow_patterns=patterns or None)
print(f"[download] {repo_id} -> {local_dir}")
EOF
}

A=$STRUCTWAM_ASSETS
for part in "${parts[@]}"; do
    case "$part" in
        cosmos)
            hf_get model nvidia/Cosmos-Predict2.5-2B "$A/models/Cosmos-Predict2.5-2B" "robot/action-cond/*" "tokenizer.pth"
            hf_get model nvidia/Cosmos-Reason1-7B "$A/models/Cosmos-Reason1-7B" ;;
        umt5)
            hf_get model Wan-AI/Wan2.1-T2V-1.3B-Diffusers "$A/models/Wan2.1-T2V-1.3B-Diffusers" "text_encoder/*" "tokenizer/*" ;;
        student)
            hf_get model xiangantao/structwam "$A/_hf_structwam" "student/step_0002000.pt" "student/text.pt" "student/appearance_stats.npz"
            mkdir -p "$A/student"
            for f in step_0002000.pt text.pt appearance_stats.npz; do ln -sfn "$A/_hf_structwam/student/$f" "$A/student/$f"; done ;;
        b2)
            hf_get model xiangantao/structwam "$A/_hf_structwam" "b2/*"
            mkdir -p "$A/checkpoints"
            ln -sfn "$A/_hf_structwam/b2" "$A/checkpoints/b2" ;;
        libero)
            hf_get dataset OpenWAM/LIBERO "$A/datasets/openwam/libero" ;;
        *) echo "unknown part: $part (cosmos umt5 student b2 libero)" >&2; exit 2 ;;
    esac
done
echo "[download] done: ${parts[*]}"
