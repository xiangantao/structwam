# V-JEPA2 single-head SSv2 baseline

For the training pool image, mount and prepared eight-GPU entry, see [POOL.md](POOL.md).

This runner freezes the official V-JEPA2 ViT-L/16 `target_encoder` and trains one
randomly initialized attentive classifier. It imports the local upstream encoder,
multiclip wrapper, classifier, transforms, decoder and optimizer; training metrics,
DDP-compatible epoch execution, checkpoints and W&B reuse HeFT's existing modules.
All new code is contained in this directory. Existing Wan sources are unchanged.

## Comparison protocol

| Setting | V-JEPA2 baseline | Existing Wan experiment |
| --- | --- | --- |
| Data/sampling | Same SSv2 manifests and per-video seed 42 | Same |
| Input | 16 frames, stride 4, two segments, 256px | Same sampled frames and augmentations |
| Encoder | Frozen ViT-L, BF16 autocast | Frozen Wan features |
| Head | One depth-4, 16-attention-head probe, 174 classes | Same architecture family |
| Training | 8 ranks x 32 videos, 20 epochs | Same |
| Optimizer | AdamW, LR 0.0003, WD 0.1, cosine LR, no warmup | Same |
| Head arithmetic | FP16 autocast and gradient scaler | Same |
| Evaluation | Separate job; 2 segments x 3 views, mean softmax | Same |
| Model selection | Primary result: epoch 20 | Use epoch 20 for Wan too |

V-JEPA2 inputs retain official ImageNet normalization, without the Wan adapter's
uint8 roundtrip. ViT-L uses its native positional encoding; no extra encoding is
added to the classifier. A clip produces `[B,2048,1024]`; two segments concatenate
to `[B,4096,1024]`. Wan uses its native 896-dimensional fused representation.
Consequently the head parameter counts and representation sizes differ; this
matches training budget and head design, not parameter count or encoder pretraining.
This is a controlled single-head comparison, not a reproduction of the published
20-hyperparameter-search result. Fix any additional evaluation epochs for both
models in advance instead of selecting separate best validation checkpoints.

`batch_size=32` counts videos per rank. `extract_batch_size=32` caps independent
clips per encoder call: two segments from 16 videos per call, two calls per full
rank batch. It does not change the global optimizer batch of 256 videos. No latent
cache or compilation is enabled for this baseline. Extraction defaults to one
lane; `--extract-lanes 2` enables two independent frozen encoder copies on fixed
threads and CUDA streams, with 32 clips per lane in this configuration.

## Local prerequisites

Use the existing `heft/.venv`. In addition to the shared classification dependencies,
the encoder requires `einops==0.8.1` (wheel `einops-0.8.1-py3-none-any.whl`).
Put official V-JEPA2 ViT-L encoder weights at `models/vjepa2/vitl.pt` in the workspace,
or pass `--model-path /absolute/path/vitl.pt`. This is the pretrained encoder file,
not a classifier or V-JEPA2.1 checkpoint. Nothing is downloaded or installed by
these scripts. Missing keys, extra keys, duplicate prefix-stripped keys and tensor
shape mismatches all fail rather than leaving randomly initialized encoder weights.

## Commands

Run from the HeFT repository root:

```bash
# CPU prerequisite check only; never starts training.
bash latent_video/classification/baselines/vjepa2/launch.sh --check

# One node, eight GPUs. No synchronous or final inline validation.
bash latent_video/classification/baselines/vjepa2/launch.sh

# Bounded test with the full training batch, sampling and LR schedule.
# Use a separate output; this is not a completed training epoch.
bash latent_video/classification/baselines/vjepa2/launch.sh \
  --output-dir ../runs/ssv2_vjepa2_smoke --smoke-steps 10

# Compare serial and parallel extraction, then measure 40 real training steps.
bash latent_video/classification/baselines/vjepa2/launch.sh \
  --output-dir ../runs/ssv2_vjepa2_smoke_2x32 --extract-lanes 2 --smoke-steps 40

# Resume the same run with the same number of ranks and identical protocol.
bash latent_video/classification/baselines/vjepa2/launch.sh \
  --resume ../runs/ssv2_vjepa2_vitl_singlehead/latest.pt

# On separate available GPUs after the epoch checkpoint exists.
NPROC_PER_NODE=8 bash latent_video/classification/baselines/vjepa2/launch.sh \
  --evaluate ../runs/ssv2_vjepa2_vitl_singlehead/epoch_0020.pt

# CPU contract tests, including a small official head with backward/update/restore.
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m pytest -q latent_video/classification/baselines/vjepa2/test_baseline.py
```

The launcher defaults to the shared `.venv`; set `PYTHON_BIN` to a prepared local
environment if desired. It does not stage the dataset, so shared-filesystem read
performance still needs measurement on the training node. Avoid scheduling this
job on GPUs already occupied by the Wan run.

`--check` checks local file availability and import discovery only. It does not
validate checkpoint tensors, CUDA execution, full dataset decoding or throughput.
Real pretrained loading and a GPU smoke run remain necessary after the weights
and dependency arrive; CPU tests do not substitute for that verification.

## Outputs

Default output: `runs/ssv2_vjepa2_vitl_singlehead/` in the workspace.

- `launch_logs/launch_*.log`: persistent startup, stdout, stderr and traceback log.
- `monitoring.jsonl`, `metrics.jsonl`, `wandb/`: offline monitoring and epoch metrics.
- `epoch_0001.pt` through `epoch_0020.pt`: immutable probe/optimizer/RNG checkpoints.
- `latest.pt`: atomic alias to the last completed checkpoint, not a validation winner.
- `config.json`, `protocol.json`: effective configuration, weight/content hashes,
  source hashes, preprocessing, parameter counts and dataset identity.
- `evaluations/epoch_0020/`: exact-coverage metrics and per-rank top-5 predictions.

Use immutable `epoch_*.pt` paths for asynchronous evaluation. Evaluation may use a
different GPU count or batch size; resume requires the same training protocol and
world size. Current shared checkpoint validation includes resolved dataset paths
and the upstream source directory: use the same mounts for training and evaluation.
Set `LOG_DIR` explicitly when directing `--output-dir` to a different location;
otherwise launch logs stay in the default run's `launch_logs` directory.

The bounded test uses the first requested batches of the original distributed
sampler, keeps the full 660-step epoch length for the optimizer schedule, and
checks parameter updates, frozen encoder gradients, DDP parameter agreement and
checkpoint restoration. It saves `smoke.pt` and `smoke_result_r*.json`. Its partial
epoch checkpoint is marked in the protocol and cannot resume a normal run. Batch
scheduling ends at the requested limit so video workers can exit normally.
Two-lane smoke tests require exact serial/parallel output agreement on the first
batch, then time both extraction modes after warmup. Step reports also include
the slowest rank's data wait and extraction time; these separate maxima need not
come from the same rank. Instrumented extraction synchronizes CUDA for timing.

The baseline CSV reader resolves each parent directory once and inspects directory
entries for symbolic links. It preserves the existing manifest identities while
avoiding repeated shared-filesystem path resolution for every video.
