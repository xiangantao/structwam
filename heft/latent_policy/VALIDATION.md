# Validation record

Housekeeping, 2026-10-11: the three completed short DDP trials below retain
readable metadata under `runs/_test_records/`; their disposable trial checkpoints
were removed. Formal training checkpoints and `.runtime` validation reports were
retained. Historical claims describe the checks performed at the recorded time.

## 2026-09-18: six extraction GPUs and one ACT learner, ready-first queue

Independent prototype `scripts/trial_async_policy.py`, not the official training runner.
Extraction uses GPUs 0,1,2,4,6,7 with 64 samples per producer, two lanes of microbatch 8;
GPU 5 runs ACT with six accumulated microbatches per optimizer update (global 384).
No DDP or disk latent cache. Real Spatial images, frozen Wan weights and ACT backward/AdamW.
Three updates / 1,152 sample exposures completed; all owned workers exited.
All 18 expected (update, producer) batch keys were consumed exactly once. The saved
40,593,095-parameter policy reloaded successfully on CPU.

The initial prototype consumed producers in fixed GPU order, causing head-of-line blocking.
It now consumes whichever member of the current logical update is ready, retaining future
update payloads separately. Per-producer credits bound prefetch. The first observed physical
GPU consumption order was 6,0,4,7,1,2; the next two were 6,0,7,4,1,2. ACT training overlaps
extraction, although each update still requires all six assigned microbatches.

Step times: 85.300, 52.153, 47.629 s. The last two average **49.891 s / 384 samples**,
or **7.697 samples/s**: learner waiting 37.838 s, host-to-device transfer 0.931 s,
ACT computation including optimizer 11.059 s. Learner peak PyTorch allocated memory
was **18.230 GiB**, excluding reserved memory and CUDA overhead. Only two post-initialization
updates were measured on shared GPUs whose competing load changed during the experiment.
This is not a controlled speedup comparison against six-GPU DDP: this prototype uses seven
GPUs, different scheduling, and a different sampler. Do not infer convergence from its loss.

Artifacts: `.runtime/async6plus1_ready_b64/{manifest.json,events.jsonl,summary.json,checkpoint}`.
The earlier fixed-order run is retained at `.runtime/async6plus1_b64` for diagnosis.
Checkpoint scope is policy and processors only, without optimizer resume state. This prototype
does not provide the official runner's full resume or W&B integration. CPU host transport and
fixed producer assignments remain optimization opportunities.

All 18 policy tests passed after the scheduling change, including ready-first ordering,
future-update retention, and duplicate-batch rejection. Ruff check and format check passed
for the modified worker, trial script and new queue tests.

## 2026-09-18: six GPUs, batch 64, two extraction lanes

Completed four official training steps on physical GPUs 0,1,2,4,6,7 with batch 64 per
rank (global 384), two independent extraction lanes, microbatch 8 per lane, compilation
disabled, BF16 ACT and zero DataLoader workers. No OOM, no fallback to batch 48.
Checkpoint `000004` was saved. Training and all extraction workers exited successfully.
Retained metadata: `runs/_test_records/libero_spatial_wan_ddp6_b64_lanes2_20260918_074633` relative to workspace.

First step including loading and per-lane warmup: 170.797 s. Subsequent steps:
129.301, 130.858, 126.290 s; mean **128.816 s**, throughput **2.981 samples/s**.
At 800,000 sample exposures, 2,084 steps extrapolate to **74.57 hours per suite**,
excluding evaluation/checkpoint overhead. Only three post-initialization steps were measured
and the GPUs were shared. This is not dedicated-hardware throughput or convergence evidence.

Five-second nvidia-smi samples, summing this run's ACT and Wan processes on each card,
gave per-card sampled peaks of **47.21–47.79 GiB**. Shorter transients may be missed.
The trainer's 19.75 GiB memory statistic only covers ACT allocated memory.
Raw config/log/GPU samples/summary are under `.runtime/ddp6_b64_lanes2*`.

The earlier single-lane microbatch-16 batch-64 run on 0,1,2,4,5,7 was deliberately
interrupted after two updates when the user redirected the experiment toward parallel
extraction. Its second step was 157.174 s. Different GPU selection, changing shared
workloads and that single measurement preclude attributing the timing difference solely
to extraction concurrency. No completed checkpoint is claimed for the interrupted run.

## 2026-09-18: batch 8 four-GPU trial

The same GPUs 0,1,2,7 completed 12 official training updates with batch 8 per rank
(global 32), extraction microbatch 8, one lane, no compilation and zero loader workers.
Checkpoint `000012` was saved; the run exited successfully and its workers exited.
Retained metadata: `runs/_test_records/libero_spatial_wan_ddp4_b8_20260918_072259` relative to workspace.

Excluding initialization, mean step time was 19.633 s (1.630 samples/s); the last five
averaged 14.178 s. Shared workloads changed during the run, so this is not a controlled
batch-size speedup measurement. At fixed 800,000 exposures the full post-initialization
mean extrapolates to 136.3 hours per suite, excluding evaluation and startup.

A live process snapshot showed ACT at 4,814 MiB and Wan at 9,512 MiB on each selected
GPU, totaling 13.990 GiB per card. This is a snapshot, not a measured allocation peak;
the training log's 3.09 GiB excludes Wan and CUDA/reserved overhead. Larger batch memory
budgets are estimates only: batch 16/extraction 16 about 20–28 GiB; batch 32/extraction 16
about 28–40 GiB; batch 64/extraction 16 about 40–60 GiB. No larger batch was tested here.
Results: `.runtime/ddp4_b8_summary.json`, config/log with the same prefix.

## 2026-09-18: four-GPU official training trial

Completed the official LeRobot training runner on physical GPUs 0,1,2,7, with Wan features,
Spatial's 432 episodes / 52,970 frames, DDP replication 4, batch 2 per rank, 20 optimizer
updates (160 sample exposures), BF16 and offline W&B. Extraction batch 2, one lane,
compilation disabled, zero DataLoader workers for this bounded integration trial.

All ranks completed without OOM or runtime errors. Checkpoint `000020` includes policy,
processors, optimizer and RNG state; reloading the policy on CPU succeeded (40,593,095
trainable parameters). Full optimizer resume has not been exercised. Worker processes exited.

First step including model initialization: 114.703 seconds. Subsequent 19 steps averaged
7.210 seconds, median 8.272, range 1.581–11.599 seconds; average throughput 1.110 samples/s.
All four GPUs were shared with other workloads, so these are not dedicated-card throughput
results. Mean data time was 0.182 seconds. Logged 1.30 GiB memory is only the policy process
and excludes the separate Wan extraction process. Loss changed 109.760 to 12.110, largely
through KL reduction; this brief trial is not evidence of task success.

Retained metadata: `runs/_test_records/libero_spatial_wan_ddp4_trial_20260918_071216` relative to workspace.
Configuration, log and summary: `.runtime/ddp4_trial.json`, `ddp4_trial.log`,
`ddp4_trial_summary.json`. This bounded trial finished; no full-budget training was launched.

## 2026-09-18: offline environment and real-weight integration

Installed the supplied, SHA256-verified offline bundle into `latent_policy/.venv`.
The existing `heft/.venv` and modified Diffusers remain unchanged. No network downloads.

| Component | Installed version |
| --- | --- |
| Python | 3.12.14 |
| PyTorch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| LeRobot / local plugin | 0.6.2 / 0.1.0, editable local sources |
| Transformers / Hugging Face Hub | 5.5.4 / 1.30.0 |
| hf-libero / robosuite / MuJoCo | 0.1.4 / 1.4.0 / 3.8.1 |
| NumPy / PyAV / PyArrow | 2.2.6 / 15.1.0 / 25.0.1 |
| W&B | 0.27.2, offline |

Full installed distribution list: `.runtime/policy-environment.txt`.
Both OpenCV distributions are required by the supplied dependency graph; headless 4.13.0.92
was installed last so the active `cv2` is 4.13.0. Reinstalling opencv-python can overwrite it.

| Check actually run | Result |
| --- | --- |
| Offline dependency resolution / `uv pip check` | All 160 distributions compatible |
| Preflight, including original Wan interpreter | `ready: true`, no issues |
| Policy tests with real LeRobot dependencies | 16 passed, none skipped |
| Ruff / launcher shell syntax | Passed |
| Local LIBERO data | 1,693 episodes, 273,465 frames, 40 tasks; dual-camera history decoded |
| LIBERO EGL simulation | Actual Spatial task reset/render/one fixed action step passed; both cameras 256x256 |
| Wan + full ACT, real data/weights | Forward, finite loss, backward, finite projection gradient, AdamW update passed |
| V-JEPA2 ViT-L + full ACT, real data/weights | Strict checkpoint load and the same update checks passed |

Smoke tests used one sample, two cameras, 16 history frames, ACT width 512 and BF16 autocast.
Wan loss 111.513313, projection gradient norm 9.335994; V-JEPA2 loss 111.478996,
projection gradient norm 7.404923. These are execution checks on an untrained policy, **not
quality comparisons**. Results are in `.runtime/smoke_wan.json` and `smoke_vjepa2.json`.
Reported allocated memory covers only the policy process, not the separate Wan worker.

Wan uses a persistent local subprocess in the original Transformers 4.57.6 / Diffusers 0.34.0
environment. CPU NumPy payloads cross the authenticated loopback connection; feature values
are transported losslessly via float32 and restored to the original dtype. Features are never
cached to disk. This resolves the Transformers 4/5 dependency conflict without upgrading SSv2.
No throughput, DDP scaling, complete learned-policy rollout, or convergence result is claimed.

The supplied v3 episode metadata omits task membership; preparation now recovers it from
frame task IDs. Spatial selection contains 432 episodes / 52,970 frames and all ten tasks.
Storage timestamps declare 10 FPS; temporal queries use those timestamps for consecutive rows,
while simulation retains official 20 Hz control. Their physical timing correspondence still
needs dataset-provenance confirmation before interpreting policy quality.

Activate paths with `source heft/latent_policy/scripts/env.sh`; run commands using
`"$LATENT_POLICY_PYTHON"`. Data is in `datasets/libero`, assets in `simulation/libero-assets`.
LIBERO config and build/cache files are private to `.runtime`.

## Historical implementation record: 2026-09-13

The remaining sections describe the earlier implementation stage, before offline installation.

## Implemented scope

The plugin reuses the local LeRobot ACT core, processors, dataset history machinery and
official runners. The external ACT extension patch is stored under `patches/` and already
applied to `repos/lerobot`. The existing SSv2 fusion implementation now calls the same
`fuse_candidates` helper as the Wan policy adapter.

No packages, weights or datasets were downloaded or installed. No GPU benchmark, real-weight
ACT training, simulation rollout, git commit or remote push was performed in this implementation.

## Checks actually completed

| Check | Result |
| --- | --- |
| `python -m pytest heft/latent_policy/tests -q` | 12 passed, 1 skipped |
| Existing SSv2 `test_online_fusion_keeps_channel_and_segment_order_and_reextracts` parameterizations | 5 passed, 19 unrelated tests deselected |
| `ruff check heft/latent_policy` | Passed |
| Shell syntax of `scripts/env.sh` | Passed |
| ACT patch reverse-application check with `--ignore-space-change` | Passed |
| `git diff --check` in heft | Passed |

The new tests cover suite membership/counts, RGB uint8/float equivalence, rejection of
normalized or malformed inputs, strict checkpoint keys/shapes, mask/checkpoint hashes,
history padding/snapshots/reset, updating history while queued actions are consumed,
camera/time position shapes, detached visual features and trainable projection gradients.

The ACT regression test compares actual original source from local LeRobot Git HEAD with the
modified source: forward outputs are equal with zero tolerance; parameter gradients pass
the specified tight tolerance. SDPA attention output/input-gradient checks pass for both
encoder and decoder layers. These use a small CPU ACT, dropout zero and synthetic inputs.
They are not a claim of bitwise equivalence under GPU BF16 or training dropout.

A synthetic full ACT step using mock Wan feature maps completed L1+KL, backward and AdamW.
The visual projection changed, and safetensors save/load reproduced evaluation actions with
zero tolerance. The frozen feature extraction itself was mocked in this check.

The dependency-light harness executes the real source definitions with a minimal config/base
stand-in to avoid importing unavailable CLI/data dependencies. The skipped module is the
real LeRobot registration/config/processor/dataset integration suite. No end-to-end LeRobot
integration success is asserted by the dependency-light results.

## Environment limitations

The read-only preflight returns `ready: false` in `heft/.venv`.

Missing distributions: draccus, datasets, gymnasium, av, pyarrow, deepdiff, jsonlines,
termcolor, hf-libero, robosuite, mujoco, cmake.

Installed versions outside this local LeRobot checkout's core declarations:

| Package | Installed | LeRobot declaration |
| --- | --- | --- |
| numpy | 2.5.1 | >=2.0, <2.3 |
| opencv-python-headless | 5.0.0.93 | >=4.9, <4.14 |
| huggingface-hub | 0.36.2 | >=1.6, <2 |
| packaging | 26.2 | >=24.2, <26 |

The current Wan runtime has Transformers 4.57.6 and modified local Diffusers 0.34.0.
LeRobot's LIBERO extra declares Transformers >=5.4,<5.6. Upgrading the active Wan environment
blindly is not an environment solution. These are actual declared-version mismatches;
whether a restricted ACT runtime can operate outside some declarations is still unverified.

## Pending real integration

1. Supply an offline compatible policy/simulation environment without breaking SSv2's runtime.
2. Supply local LeRobot-format LIBERO data and verify task descriptions, 20 Hz metadata,
   camera orientation, 8-D state and 7-D action conventions.
3. Generate/audit each suite's episode list and the two representation configs.
4. Validate the actual V-JEPA2 checkpoint with strict loading and run both actual extractors.
5. Run the real integration tests, then a small real-data training and LIBERO rollout smoke test.
6. Measure ACT memory and throughput before selecting extraction lanes/batches or launching DDP.

All-token comparison deliberately retains 6,272 Wan versus 4,096 V-JEPA2 visual tokens for two
cameras. Camera/history input, ACT architecture and sample budgets match; token/compute budgets
and input-projection parameter counts do not. No task-success or throughput result exists yet.
