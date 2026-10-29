#!/bin/bash
# Arm B: same recipe as the direct (teacher) arm, one change -- per-episode feature seeds.
#   config: configs/generated/direct_b/libero_spatial_wan_jepa_perclip.json
#   A differs from this only in job_name / output_dir / policy.feature_seed / feature_seed_mode.
# Launch on the 8-GPU training machine. Nothing here writes outside $W and /tmp.
set -euo pipefail

W=/increase_kairos_vepfs/increase/liwenhao/agent/2337888765
LP=$W/heft/latent_policy
CFG=$LP/configs/generated/direct_b/libero_spatial_wan_jepa_perclip.json

# ---- environment -------------------------------------------------------------------
# The venvs' editable .pth files point at a moved mount (/kairos_vepfs_volc/...), so
# PYTHONPATH must be explicit. LATENT_WAN_JEPA_WORKER selects the per-clip-seed worker;
# without it the original worker is used and the run dies on the 3-tuple request.
export PYTHONPATH="$LP/src:$W/heft:$W/heft/src:$W/repos/lerobot/src"
export LATENT_POLICY_PYTHON="$LP/.venv/bin/python"
export LATENT_WAN_PYTHON="$W/heft/.venv/bin/python"
export LATENT_WAN_JEPA_WORKER="$LP/scripts/wan_jepa_worker_seeded.py"
export LATENT_ACT_TMPDIR="${LATENT_ACT_TMPDIR:-/tmp/heft-latent-act}"
export TMPDIR="$LATENT_ACT_TMPDIR"
export LIBERO_CONFIG_PATH="$LP/.runtime/libero"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
mkdir -p "$LATENT_ACT_TMPDIR" "$W/logs"

LOG="$W/logs/arm_b_perclip_$(date +%Y%m%d_%H%M%S).log"
echo "workspace : $W"
echo "config    : $CFG"
echo "log       : $LOG"
echo "worker    : $LATENT_WAN_JEPA_WORKER"
echo "gpus      : $(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
echo

# ---- sanity: the per-clip worker and the routing must both be present -----------------
[ -f "$LATENT_WAN_JEPA_WORKER" ] || { echo "missing worker: $LATENT_WAN_JEPA_WORKER"; exit 1; }
grep -q 'feature_seed_mode.*per_clip' "$LP/src/lerobot_policy_latent_act/encoders/__init__.py" \
  || { echo "per_clip routing missing in encoders/__init__.py"; exit 1; }
python3 -c "import json;d=json.load(open('$CFG'));assert d['policy']['feature_seed_mode']=='per_clip';assert d['steps']==12500;assert d['batch_size']==8;assert d['parallelism']['dp_replicate']==8;print('config OK: per_clip, 12500 steps, batch 8 x 8 ranks')"

# ---- read-only preflight --------------------------------------------------------------
"$LATENT_POLICY_PYTHON" "$LP/scripts/preflight.py" --config "$CFG" 2>&1 | tee -a "$LOG"
echo "preflight exit: ${PIPESTATUS[0]}"
echo

# ---- 8-GPU training -------------------------------------------------------------------
set -x
"$LATENT_POLICY_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$LP/scripts/run.py" train --config_path="$CFG" \
  --output_dir="$W/runs/libero_spatial_wan_jepa_direct_perclip_20260930" 2>&1 | tee -a "$LOG"
