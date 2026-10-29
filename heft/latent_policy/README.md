# Frozen video representations with ACT on LIBERO

The frozen VACE attention motion-readout protocol is documented in
[`VACE_ATTENTION_MOTION_READOUT_PIPELINE.md`](VACE_ATTENTION_MOTION_READOUT_PIPELINE.md),
with a Chinese review version in
[`VACE_ATTENTION_MOTION_READOUT_PIPELINE_ZH.md`](VACE_ATTENTION_MOTION_READOUT_PIPELINE_ZH.md).

This directory adapts the local LeRobot ACT implementation. It does not implement a new
policy trainer, action decoder, CVAE objective, video dataset, or LIBERO evaluator.

The bounded distilled ACT256 suite contract and the four generated configs are documented in
[`DISTILLED_LIBERO.md`](DISTILLED_LIBERO.md).

## Experiment contract

- Train one policy per suite: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`.
  Repeat for Wan and V-JEPA2: eight runs before additional seeds.
- Both use two 256x256 RGB cameras, the 16 consecutive observations ending at the current
  step, an 8-D current robot state, and 7-D actions. Initial history repeats the first frame.
  Future action targets are allowed; future observations are not.
- Start with ACT chunk size 16, executing 8 actions before replanning. These are explicit
  experiment choices, not claimed LIBERO-optimal ACT hyperparameters.
- Both use the same ACT CVAE, encoder/decoder, 512-D visual projection output, L1 + KL
  objective, task one-hot path, optimizer, data selection, sample budget and evaluation seeds.
  ACT's action-distribution latent is distinct from the visual representation.
- Wan keeps the existing five candidates: 128 + 128 + 256 + 256 + 128 = 896 channels.
  Noise schedules, global 256-channel mask, modified causal VAE, 16-to-17 padding and
  B15 capture come directly from `latent_video`. No new Wan feature definition is introduced.
- Frozen features are computed online. They are not persisted as a latent cache.
- Training does not run simulation evaluation by default; checkpoints can be evaluated
  separately. W&B is offline and Hub upload/download is disabled in the launcher.

## What is reused

| Component | Existing source relative to workspace root |
| --- | --- |
| ACT config, optimizer and action target indices | `repos/lerobot/src/lerobot/policies/act/configuration_act.py` |
| ACT CVAE, Transformer, losses and action queue | `repos/lerobot/src/lerobot/policies/act/modeling_act.py` |
| State/action normalization and checkpointable processors | `repos/lerobot/src/lerobot/policies/act/processor_act.py` |
| LIBERO camera rotation and robot state conversion | `repos/lerobot/src/lerobot/processor/env_processor.py` |
| Camera history queue and padding | `repos/lerobot/src/lerobot/policies/utils.py`, `policies/pi05/memory.py` |
| Dataset history queries and episode boundary clamping | `repos/lerobot/src/lerobot/datasets/factory.py`, `dataset_reader.py` |
| Wan feature extraction, fusion and execution lanes | `heft/latent_video/extractor.py`, `classification/encoder.py`, `classification/execution.py` |
| V-JEPA2 ViT-L and checkpoint format | `vjepa2/src/models/vision_transformer.py`, `evals/video_classification_frozen/modelcustom/vit_encoder_multiclip.py` |
| Training, DDP, optimizer/checkpoint resume | `repos/lerobot/src/lerobot/scripts/lerobot_train.py` |
| Per-task rollout success metrics | `repos/lerobot/src/lerobot/scripts/lerobot_eval.py`, `envs/libero.py` |

The local ACT patch adds `_make_model`, `_build_visual_backbone`, `_visual_tokens` extension
methods and an opt-in `need_attention_weights` switch. Default ACT still uses ResNet and
requests attention weights. LatentACT disables unused attention weights for SDPA and uses
PyTorch's non-reentrant activation checkpoint wrapper on ACT encoder layers for both backbones.

`patches/lerobot-act.patch` preserves the necessary external-repository modification in this
repository. It was created against local LeRobot commit
`b6ec0060779550c0a157ae34feb89e0cf86012a8`. It is already applied in this workspace.
On a matching fresh local checkout, apply it once with:

```bash
git -C repos/lerobot apply --check --ignore-space-change ../../heft/latent_policy/patches/lerobot-act.patch
git -C repos/lerobot apply --ignore-space-change ../../heft/latent_policy/patches/lerobot-act.patch
```

## Fairness and input details

Wan outputs `[B,16,896,14,14]` per camera: 6,272 total visual tokens for two cameras.
V-JEPA2 ViT-L outputs `[B,8,1024,16,16]` per camera: 4,096 total visual tokens.
The raw input observations and ACT architecture match, but the token counts, projection
parameter counts and compute costs do not. This is an all-token representation comparison,
not a claim of equal token/compute budgets. Do not silently pool either time axis to make them equal.

All RGB inputs are checked and converted to the same CPU uint8 pixels before either backbone.
LeRobot's dataset float conversion is reversed by rounding. Wan applies its existing video
preprocessor; V-JEPA2 applies official ImageNet mean/std. Input size mismatches fail rather
than resizing back to the earlier 480x832 setting. No image augmentation is enabled initially.
This common CPU pixel contract is intentionally explicit; it can incur device round trips
under Accelerate and has not been benchmarked as an ACT input pipeline.

ACT spatial sinusoidal positions are reused. Both policies also receive fixed camera codes
and the same 16 input-time codes from LeRobot's temporal helper. For V-JEPA2's tubelets,
each adjacent pair of input-time codes is averaged. This adds no trainable positional layer.

Task descriptions identify one of the suite's 10 IDs; they are not encoded as language.
The one-hot uses ACT's existing `observation.environment_state` token and replaces any incoming
value at that key. No privileged simulator object state is consumed. A config's task list must
come from the actual local LIBERO benchmark, or a verified explicit task manifest.

## Environment status

The offline bundle has been installed into `heft/latent_policy/.venv`. Source the launcher
environment and run the read-only dependency check without allocating a GPU:

```bash
source heft/latent_policy/scripts/env.sh
"$LATENT_POLICY_PYTHON" heft/latent_policy/scripts/preflight.py
```

Wan retains the existing `heft/.venv`, Transformers 4.57.6 and modified Diffusers 0.34.0.
The policy/simulator uses its own Transformers 5 environment. `LATENT_WAN_PYTHON` selects a
persistent local worker for frozen Wan extraction; the worker loads weights once and returns
CPU feature payloads to ACT. This is online extraction, with no latent disk cache. CPU transfers
add overhead and this execution path has not been throughput tuned. V-JEPA2 runs in the policy
process. The old SSv2 environment has not been upgraded.

Data lives in `datasets/libero`, simulation assets in `simulation/libero-assets`. LIBERO config
and temporary/cache files live under this directory's `.runtime`. `env.sh` selects these paths
and enforces offline Hugging Face and W&B modes. See `VALIDATION.md` for actual test coverage.

## Prepare suite-specific configs

Commands below assume the required offline dependencies and a complete local LeRobot-format
LIBERO dataset have been supplied. They perform no package/model/data downloads.
Use the actual Python environment with the compatible dependency set.

```bash
source heft/latent_policy/scripts/env.sh
LATENT_PYTHON="$LATENT_POLICY_PYTHON"
```

The template is deliberately not directly trainable: its task list and episode list are empty.
Preparation loads local LeRobot metadata, matches the official suite's task descriptions,
requires demonstrations for all ten tasks, and writes an episode/count/budget audit next to
the generated config. It does not assume the four suites occupy contiguous episode ranges.
This dataset declares 10 FPS storage timestamps. History/action indexing follows consecutive
dataset rows using that metadata; simulator control remains the official 20 Hz. The audit
records both rates. Storage metadata alone does not prove the demonstration's physical
capture rate; that timing correspondence needs verification before interpreting policy quality.
When v3 episode metadata omits tasks, membership is recovered from frame `task_index` columns.

Example for Wan, 8 processes with per-process batch 8:

```bash
"$LATENT_PYTHON" "$LATENT_POLICY_ROOT/scripts/run.py" prepare \
  --dataset-root "$LATENT_WORKSPACE/datasets/libero" \
  --suite libero_spatial --encoder wan \
  --template "$LATENT_POLICY_ROOT/configs/libero_template.json" \
  --output "$LATENT_POLICY_ROOT/configs/generated/spatial_wan.json" \
  --run-dir "$LATENT_WORKSPACE/runs/libero_spatial_wan" \
  --wan-model-path "$LATENT_WORKSPACE/models/Wan2.1-T2V-1.3B-Diffusers" \
  --channel-mask "$LATENT_HEFT_ROOT/reports/scannet_channels/73f4e810824b58cb/global_256/masks.json" \
  --world-size 8 --batch-size 8 --sample-budget 800000
```

For V-JEPA2 use `--encoder vjepa2`, distinct output/run paths, and replace the Wan arguments with
`--vjepa-root "$LATENT_WORKSPACE/vjepa2" --vjepa-checkpoint /absolute/path/to/verified/vitl.pt`.
The local checkpoint is `models/vjepa2/vitl.pt`. The adapter rejects missing/mismatched
backbone keys instead of falling back to randomly initialized tensors.
Repeat for all four suite names. An optional `--task-manifest /absolute/path/tasks.json` can
provide `{ "libero_spatial": ["official task description", ...], ... }` without importing LIBERO
during preparation; the caller must verify this manifest against the benchmark.

800,000 sample exposures comes from the generic ACT example's 100,000 steps at batch 8.
At global batch 64 it means 12,500 optimizer steps; at global batch 256 it means 3,125 steps.
This is an initial comparison budget, not a convergence guarantee. The generator records the
actual rounded exposure count and suite-specific approximate epochs. It assumes accumulation 1.

## Train, resume, and evaluate

After dependency and data checks pass, the launcher only registers this plugin and calls the
official runner. Extraction starts lazily on the first visual forward on each rank's own device.

```bash
"$LATENT_PYTHON" "$LATENT_POLICY_ROOT/scripts/preflight.py" \
  --config "$LATENT_POLICY_ROOT/configs/generated/spatial_wan.json"
"$LATENT_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$LATENT_POLICY_ROOT/scripts/run.py" train \
  --config_path="$LATENT_POLICY_ROOT/configs/generated/spatial_wan.json"
```

Start with the configured extraction batch 8 and one lane until real ACT memory/throughput
has been measured. Wan can then use `--policy.extract_lanes=2`, `--policy.extract_batch_size=32`,
and `--policy.compile_blocks=true`; these are available options, not an untested memory guarantee.
V-JEPA2 rejects the Wan-specific lane/compile options.

Resume through the official training config saved with the checkpoint:

```bash
"$LATENT_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$LATENT_POLICY_ROOT/scripts/run.py" train \
  --config_path=/absolute/path/checkpoints/STEP/pretrained_model/train_config.json --resume=true
```

For separate evaluation of a saved suite policy:

```bash
"$LATENT_PYTHON" "$LATENT_POLICY_ROOT/scripts/run.py" eval \
  --policy.path=/absolute/path/checkpoints/STEP/pretrained_model \
  --env.type=libero --env.task=libero_spatial \
  --env.observation_height=256 --env.observation_width=256 \
  --env.init_states=true --env.max_parallel_tasks=1 \
  --eval.n_episodes=50 --eval.batch_size=1 --seed=42
```

Use the matching suite and identical seeds/initial states for both backbones. Rollouts report
per-task success and aggregated suite success; offline action loss alone is not task success.
The history implementation follows LeRobot's synchronized rollout reset contract. It is not an
independent-reset vector policy server: call `reset()` at episode boundaries, and keep
`max_parallel_tasks=1` so concurrent tasks do not share action/history state.

Policy checkpoints contain the trainable ACT weights and serialized configuration. Frozen
encoder services are intentionally outside the `nn.Module` tree, so they are neither optimized
nor duplicated in each checkpoint. The same external encoder weights, mask and source trees
must remain available on resume/evaluation. Generated configs pin the mask/V-JEPA2 file hashes.

## Verification

```bash
source heft/latent_policy/scripts/env.sh
"$LATENT_POLICY_PYTHON" -m pytest heft/latent_policy/tests -q
heft/.venv/bin/ruff check heft/latent_policy
```

The dependency-light tests execute actual ACT/history source definitions with a minimal policy
base/config stand-in. They cover the visual extension's numerical regression against the local
LeRobot Git baseline, SDPA output/gradient equivalence, suite selection, pixel contracts,
strict checkpoint loading, rollout queues, and visual projection/state-dict boundaries.
They do not substitute for real LeRobot integration. `test_integration.py` uses real imports
and is explicitly skipped when the required offline dependencies are absent.
Real-weight smoke checks can be run explicitly with `scripts/smoke_local.py --encoder wan`
or `--encoder vjepa2` using the policy Python. These perform one real-data update, not a
training benchmark. See `VALIDATION.md` for short DDP and asynchronous throughput trials;
convergence remains unmeasured.

The independent `scripts/trial_async_policy.py` prototype accepts `--config` (a generated
training JSON), `--output` (a new directory), `--extract-gpus 0,1,2,4,6,7`,
`--learner-gpu 5`, `--batch-size 64` and `--steps 3`, using the policy Python after
sourcing `scripts/env.sh`. Six frozen producers feed one ACT learner without DDP.
Ready batches are consumed within fixed logical updates, with bounded prefetch and no
latent disk cache. This is a short throughput experiment: checkpoints omit optimizer resume
state, and metrics go to JSONL rather than W&B. It does not replace the official runner.

## Direct Wan+V-JEPA2 mode

`encoder="wan_jepa"` runs the promoted representation online without the distilled
student checkpoint. Each camera's 16-frame history is passed to the existing
`latent_video.wan_jepa.CombinedExtractor`: Wan's original five groups are extracted
at the native 16x16 grid, the odd Wan time positions are selected to match JEPA's eight
tubelets, and the fixed supervised JEPA projection is concatenated with Wan:
`[8,896,16,16] + [8,256,16,16] -> [8,1152,16,16]`. The ACT 1x1 input projection
is the only representation mapping trained by ACT; Wan, V-JEPA2, and the fixed
operators remain frozen. No latent cache is written.

A prepared local Spatial config is:

```text
configs/generated/direct/libero_spatial_wan_jepa.json
```

It uses the same dataset, ACT dimensions, optimizer, sample budget, and task conditioning
as the distilled comparison. The isolated worker uses `wan_runtime_python` and requires the
local Wan checkpoint, V-JEPA2 `vitl.pt`, `vjepa_root`, and
`latent_video/wan_jepa/artifacts`. Run the read-only preflight before training:

```bash
heft/latent_policy/.venv/bin/python heft/latent_policy/scripts/preflight.py \
  --config heft/latent_policy/configs/generated/direct/libero_spatial_wan_jepa.json
```

The real one-sample check is:

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline \
PYTHONPATH=heft/latent_policy/src:heft/diffusers/src:heft/src:. \
heft/latent_policy/.venv/bin/python -u heft/latent_policy/scripts/smoke_local.py \
  --encoder wan_jepa
```
