#!/usr/bin/env bash
# StructWAM step 24000 on LIBERO-10, 10 video + 10 action denoising steps (sync), DiT cache off.
#   bash run_libero10.sh smoke   one trial of task 0 against the smoke server on port 8990
#   bash run_libero10.sh full    managed servers on NPUs 0-7, 10 tasks x 50 trials
# Protocol otherwise as benchmarks/libero (seed 42, settle 30, max 700 steps, horizon 10), except
# MuJoCo 3.8.1 from vace-lerobot instead of the pinned 3.3.2.
set -euo pipefail
R=/home/ma-user/work/2337888765
W=$R/workspace
EV=$W/runs/libero10_structwam_s24000_v10
mode="${1:?smoke | full}"
common=(
    --ckpt-dir "$W/runs/structwam_step24000_eval_ckpt" --ckpt-name checkpoint_step_24000.safetensors
    --server-python "$EV/server_python.sh" --libero-python "$EV/libero_python.sh"
    --libero-path "$R/envs/vace-lerobot/lib/python3.11/site-packages"
    --policy-config "$W/StructWAM/benchmarks/libero/policy_config.yml"
    --suites long --render-gpus 0 --denoise-steps 10 --inference-horizon 10
    --compile-enabled false --extra-deploy-arg optimization.dit_cache.enabled=false
    --client-max-attempts 6
)
case "$mode" in
    smoke) extra=(--gpus 0 --base-port 8990 --server-mode external --smoke --output-dir "$EV/smoke2") ;;
    full) extra=(--gpus 0,1,2,3,4,5,6,7 --replicas-per-gpu 1 --base-port 8920 --output-dir "$EV/results") ;;
    *) echo "mode: smoke | full" >&2; exit 2 ;;
esac
cd "$W/StructWAM"
nohup "$R/envs/openwam-npu/bin/python" "$EV/run_eval.py" "${common[@]}" "${extra[@]}" \
    > "$EV/${mode}_launcher.log" 2>&1 < /dev/null &
echo "$mode launcher pid $!"
