#!/usr/bin/env bash
# StructWAM step 24000 on LIBERO-plus with the official one-rollout protocol (1 trial per task,
# seed 10000, settle 30, 600/700 steps, MuJoCo 3.3.2 in envs/libero-plus), 10 video + 10 action
# denoising steps (sync), inference horizon 10, DiT cache off, two policy servers per NPU.
#   bash run_libero_plus.sh smoke      one task on NPU 0
#   bash run_libero_plus.sh sample10   10% of every suite's tasks (--task-sample-seed 42), NPUs 0-7
#   GPUS=0,2,3 bash run_libero_plus.sh sample10   only the listed NPUs (others busy)
#   REPLICAS=1 ...   servers per NPU (default 1: LIBERO-plus servers reach ~43 GB, two OOM a 61 GB NPU)
#   CLEAN=1 ...      send the original LIBERO instruction instead of task.language with the
#                    perturbation ids appended (libero_plus_python_clean.sh); output dir gets _clean
# Rerunning the same mode resumes into the same output directory (the NPU list may change).
set -euo pipefail
R=/home/ma-user/work/2337888765
W=$R/workspace
PV=$W/runs/libero_plus_structwam_s24000_v10
mode="${1:?smoke | sample10}"
client_python="$R/envs/libero-plus/bin/python" suffix=""
if [[ "${CLEAN:-0}" == 1 ]]; then
    client_python="$PV/libero_plus_python_clean.sh" suffix="_clean"
fi
common=(
    --ckpt-dir "$W/runs/structwam_step24000_eval_ckpt" --ckpt-name checkpoint_step_24000.safetensors
    --server-python "$W/runs/libero10_structwam_s24000_v10/server_python.sh"
    --libero-python "$client_python" --libero-path "$W/repos/LIBERO-plus"
    --policy-config "$W/StructWAM/benchmarks/libero-plus/policy_config.yml"
    --suites spatial,object,goal,long --render-gpus 0 --denoise-steps 10 --inference-horizon 10
    --compile-enabled false --extra-deploy-arg optimization.dit_cache.enabled=false
    --client-max-attempts 6
)
case "$mode" in
    smoke) extra=(--smoke --gpus 0 --replicas-per-gpu 1 --base-port 9100 --output-dir "$PV/smoke$suffix") ;;
    sample10) extra=(--task-sample-ratio 0.1 --gpus "${GPUS:-0,1,2,3,4,5,6,7}" --replicas-per-gpu "${REPLICAS:-1}"
                     --base-port 9000 --output-dir "$PV/sample10$suffix") ;;
    *) echo "mode: smoke | sample10" >&2; exit 2 ;;
esac
cd "$W/StructWAM"
nohup "$R/envs/openwam-npu/bin/python" "$W/StructWAM/benchmarks/libero-plus/run_all_suites.py" \
    "${common[@]}" "${extra[@]}" > "$PV/${mode}${suffix}_launcher.log" 2>&1 < /dev/null &
echo "$mode launcher pid $!"
