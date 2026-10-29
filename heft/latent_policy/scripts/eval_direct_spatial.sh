#!/usr/bin/env bash
# Evaluate the direct Wan+JEPA ACT checkpoint on LIBERO-Spatial with the same protocol as
# runs/libero_spatial_distilled_retry1_eval_012500 (50 episodes/task, batch 1, seed 42, init states).
# Tasks run in parallel, one process per task; lerobot_eval seeds every task from --seed, so the
# episodes match a single-process run. Results are merged into $OUT/eval_summary.json.
W=/increase_kairos_vepfs/increase/liwenhao/agent/2337888765
CKPT=${CKPT:-$W/runs/libero_spatial_wan_jepa_direct/checkpoints/012500/pretrained_model}
OUT=${OUT:-$W/runs/libero_spatial_wan_jepa_direct_eval_012500}
source $W/heft/latent_policy/scripts/env.sh
mkdir -p "$OUT"
pids=()
for T in 0 1 2 3 4 5 6 7 8 9; do
  CUDA_VISIBLE_DEVICES=$((T % 8)) "$LATENT_POLICY_PYTHON" "$LATENT_POLICY_ROOT/scripts/run.py" eval \
    --policy.path="$CKPT" \
    --env.type=libero --env.task=libero_spatial --env.task_ids="[$T]" \
    --env.observation_height=256 --env.observation_width=256 \
    --env.init_states=true --env.max_parallel_tasks=1 \
    --eval.n_episodes=50 --eval.batch_size=1 --seed=42 \
    --output_dir="$OUT/task_$T" > "$OUT/task_$T.log" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
"$LATENT_POLICY_PYTHON" - "$OUT" <<'PY'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1]); per_task = {}; n = s = 0
for t in range(10):
    f = out / f"task_{t}" / "eval_info.json"
    if not f.exists():
        per_task[t] = None; continue
    o = json.loads(f.read_text())["overall"]
    per_task[t] = o["pc_success"]; n += o["n_episodes"]; s += o["pc_success"] * o["n_episodes"] / 100
summary = {"per_task_pc_success": per_task, "n_episodes": n, "pc_success": 100 * s / n if n else None}
(out / "eval_summary.json").write_text(json.dumps(summary, indent=2)); print(summary)
PY
