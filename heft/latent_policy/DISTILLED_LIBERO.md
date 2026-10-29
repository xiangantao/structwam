# Bounded distilled ACT suites on LIBERO

This directory contains four independent ACT suite configurations for the frozen distilled
student. The launcher prepares one suite at a time and then runs a read-only preflight; it
never starts training or a suite-wide job list.

The frozen student checkpoint is:

```text
runs/wan_jepa_distillation_libero_equal6/step_0007560.pt
sha256: 3d537902cab56d7f223a91884459a61f5a2e2544cddafab37a18fb4871350c52
```

The configs use `heft-distill` as `distilled_root`, `vjepa2` as `vjepa_root`, and the
existing `heft/.venv/bin/python` as `distilled_runtime_python`. The runtime is kept separate
from the policy venv because the student source has its own torch/Transformers stack.

## Frozen feature and ACT contract

The student emits its fused feature map as `[B, 8, 1152, 16, 16]` for each camera. ACT sees
all eight temporal planes and all `16x16` spatial locations from both cameras. The adapter is
one trainable `1x1` projection from 1152 channels to ACT's `dim_model=256`; no shared student
latent `z`, PCA, token pooling, or channel refit is used. The student weights remain outside
the ACT module and are not written into ACT checkpoints.

Every config fixes two `256x256` RGB cameras, a causal 16-frame history, an 8-D current robot
state, 7-D actions, ACT chunk size 16, and eight executed actions before replanning. ACT uses
8 heads and the local ACT defaults for the remaining transformer dimensions. Task identity is
a ten-way suite-local one-hot. The dataset declares 10 FPS storage timestamps; simulator
rollouts retain the LIBERO 20 Hz control contract.

The local v3 dataset has the existing `train: 0:1693` split and 40 tasks. Selection is by the
actual per-frame task assignments and the episode metadata, so preparation does not rely on an
assumed episode range. The resulting suite counts are:

| Suite | Task indices | Episode IDs | Episodes | Frames | Config |
| --- | --- | --- | ---: | ---: | --- |
| `libero_10` | 0--9 | 0--378 | 379 | 101,469 | `configs/generated/distilled/libero_10.json` |
| `libero_goal` | 10--19 | 379--806 | 428 | 52,042 | `configs/generated/distilled/libero_goal.json` |
| `libero_object` | 20--29 | 807--1260 | 454 | 66,984 | `configs/generated/distilled/libero_object.json` |
| `libero_spatial` | 30--39 | 1261--1692 | 432 | 52,970 | `configs/generated/distilled/libero_spatial.json` |

Each config contains the ten official task descriptions in its `policy.task_names`, its own
selected episode list, its own `job_name`, and its own run directory (`runs/<suite>_distilled`).
The adjacent `.audit.json` records the task counts, frame total, storage/control rates, sample
budget, and rounded step count.

## Prepare one suite

Source the offline environment first. The environment deliberately puts Python's temporary
files and DataLoader IPC sockets under `/tmp/heft-latent-act`; do not override `TMPDIR` with the
long workspace path, because `multiprocessing.resource_sharer` can then fail with `AF_UNIX path
too long` before the first batch. The dataset, checkpoints and run outputs remain on the workspace.

Source the offline environment first. The default is a conservative per-process ACT batch of 8,
world size 8 (`global_batch=64`), extraction microbatch 4, and 800,000 sample exposures
(`12,500` optimizer steps). Batch, world size, and sample budget are CLI options; these defaults
are starting settings and carry no unmeasured VRAM or convergence guarantee.

```bash
source heft/latent_policy/scripts/env.sh

# Existing generated config; this command selects one suite and does not train.
"$LATENT_POLICY_PYTHON" "$LATENT_POLICY_ROOT/scripts/preflight.py" \
  --config "$LATENT_POLICY_ROOT/configs/generated/distilled/libero_spatial.json"
```

To prepare a new output path from the local metadata, select exactly one suite:

```bash
"$LATENT_POLICY_PYTHON" "$LATENT_POLICY_ROOT/scripts/launch_distilled_libero.py" \
  --suite libero_spatial \
  --output "$LATENT_POLICY_ROOT/configs/generated/distilled/libero_spatial.local.json" \
  --run-dir "$LATENT_WORKSPACE/runs/libero_spatial_distilled_local"
```

Replace `libero_spatial` in that command with one of `libero_object`, `libero_goal`, or
`libero_10` for the other suites. The required `--suite` argument prevents accidental
all-suite launches. The command reads only local files, records the checkpoint hash, writes a
config and audit, and runs preflight. It does not download, copy, stage, or modify the source
dataset/checkpoint.

The four generated local configs are:

```text
heft/latent_policy/configs/generated/distilled/libero_spatial.json
heft/latent_policy/configs/generated/distilled/libero_object.json
heft/latent_policy/configs/generated/distilled/libero_goal.json
heft/latent_policy/configs/generated/distilled/libero_10.json
```

## Optional bounded ACT check

After preflight, `check_distilled_act.py` performs one real local LIBERO update and a policy
checkpoint round-trip. Give it a fresh output directory; it is not a training run or a
throughput benchmark.

```bash
"$LATENT_POLICY_PYTHON" "$LATENT_POLICY_ROOT/scripts/check_distilled_act.py" \
  --config "$LATENT_POLICY_ROOT/configs/generated/distilled/libero_spatial.json" \
  --output "$LATENT_WORKSPACE/runs/check_distilled_spatial" \
  --device cuda:0
```

## Explicit training command

Training is intentionally separate from preparation. Launch one selected suite only after its
preflight and bounded check pass; the other three are not started automatically. For example:

```bash
"$LATENT_POLICY_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$LATENT_POLICY_ROOT/scripts/run.py" train \
  --config_path="$LATENT_POLICY_ROOT/configs/generated/distilled/libero_spatial.json"
```

Use the matching generated config to run another suite. W&B and Hugging Face access are offline,
Hub upload is disabled, evaluation during training is disabled, and `max_parallel_tasks=1` is
kept for the stateful rollout history. No formal training run or convergence result is implied
by these prepared configs.

## Separate rollout evaluation

Evaluation consumes a trained ACT policy directory. Set `--policy.path` to that checkpoint's
`pretrained_model` directory and keep `--env.task` equal to the suite encoded by the policy
checkpoint; the generated config itself is not a checkpoint. For a trained spatial policy:

```bash
"$LATENT_POLICY_PYTHON" "$LATENT_POLICY_ROOT/scripts/run.py" eval \
  --policy.path="$LATENT_WORKSPACE/runs/libero_spatial_distilled/checkpoints/STEP/pretrained_model" \
  --env.type=libero --env.task=libero_spatial \
  --env.observation_height=256 --env.observation_width=256 \
  --env.init_states=true --env.max_parallel_tasks=1 \
  --eval.n_episodes=50 --eval.batch_size=1 --seed=42
```

Use the corresponding run directory and suite name for the other three policies.


## Local acceptance on 2026-09-25

Final source passed 21 regression tests (configuration serialization, ACT forward/backward,
history and positions, camera/chunk ordering, inference tensor boundaries and checkpoint
roundtrips). All four generated configs decoded through the official LeRobot
`TrainPipelineConfig`, and their episode sets are disjoint (1,693 total).

The final adapter was checked on one real `libero_spatial` observation with both cameras,
using the frozen `step_0007560.pt` student. ACT produced `[1,16,7]` actions. The trainable
projection received a finite nonzero gradient and updated; the student was absent from
policy parameters/checkpoint. Saved/restored ACT predictions matched exactly. The worker
exited afterward. Results:

`heft/latent_policy/.runtime/distilled_act_acceptance_20260925_final/result.json`

This was one local GPU update with extraction microbatch 1. It does not validate the default
batch-8 capacity, eight-rank DDP, long training stability, or simulation success. The log's
parent peak memory excludes the student subprocess and must not be quoted as total GPU
memory. No formal suite training or simulator success evaluation has been launched.
