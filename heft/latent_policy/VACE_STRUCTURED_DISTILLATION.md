# VACE Structured Distillation: Past Video and Instructions to Observed and Future Matrices

This is a new student pipeline. It does not modify the earlier six-group Wan/JEPA
distillation or ACT adapter. The default `shared` run jointly trains five output
heads on ONE V-JEPA2 backbone, language-fusion stack and observed/future decoder
pair. Individual head modes remain available for ablations.

Mainline update, 2026-10-10: the future teacher uses ONE UniPC update from
GT-noised sigma0.801952 to the original 13-update endpoint sigma0.499667, then
a conditional readout forward. The original 13-update recipe is an explicit
reference. Online extraction is the default; the earlier TOS cache plan is archived.

## Input Boundary

The student signature is:

```python
student(observed_rgb, text_embeddings, text_mask)
```

It accepts exactly 33 past uint8 RGB frames `[B,33,3,256,256]`. It has no future
RGB, future latent, target, action or simulator-state input. The 33rd observed
frame is repeated once to complete a width-2 V-JEPA2 tubelet; no frame is dropped
and no future frame is used as padding.

Frozen UMT5-XXL produces `[B,512,4096]` word tokens. The text cache is the only
default cache. It includes the 40 actual LIBERO task instructions and the
negative generation prompt. New instructions must be encoded with the same
pretrained tokenizer and encoder. This is not a one-hot task id.

## Student Architecture

- Original local V-JEPA2 ViT-L/16 target-encoder weights, fully trainable.
- Trainable 4096-to-1024 text projection and residual text cross-attention
  after the final four backbone blocks. Visual positions query text words.
- Native video memory remains a full `[17,16,16,1024]` field at the default
  resolution. No global visual pooling or PCA is applied.
- Separate observed and future decoders. Each has two shifted-window spatial
  plus temporal attention blocks and cross-attention to past visual tokens
  and text. The future query contains spatial/time position, a learned query
  and the last past field; it never contains future pixels.
- Each of the five output heads has two more spatial/temporal Transformer
  blocks and residual MLP projections `1024 -> 2048 -> C` plus a linear skip.
- The two K-partition heads, tracking Q/K head, semantic head and depth head have
  separate parameters. They read the same decoded observed/future fields; the
  backbone is encoded only once per student forward. Losses from all heads
  backpropagate into this shared representation.
- Observed and future use the same weights within each head.
- 32x32 is a learned teacher-aligned readout grid; upsampling initializes it
  from the native 16x16 student grid and does not create native image detail.

## Five Shared-Backbone Outputs

`T_o=12`, `T_f=13`, `G=1024`, `C=128`, `L=512`. A batch output is:

```text
observed: all five heads' matrices on [B,T_o,G,...]
future:   all five heads' matrices on [B,T_f,G,...]
timeline: native times, nominal RGB support, valid masks and anchor mappings
```

| Task | Student matrices, in BOTH branches | Frozen teacher readout |
| --- | --- | --- |
| `seg_l22h9` | unit `seg_k [B,T,G,128]` | normalized pre-RoPE L22H9 K |
| `seg_l15h7` | unit `seg_k [B,T,G,128]` | normalized pre-RoPE L15H7 K |
| `tracking` | unit `track_q, track_k [B,T,G,128]` | normalized post-RoPE L15H4 Q/K |
| `semantic` | `text_attention [B,T,G,512]` | L22H4 text-attention distribution |
| `depth` | `depth [B,T,G]` | `(-K[L16,H9,52]-4.34375)/1.65625` |

In `shared` mode the flat branch keys are `seg_k_l22h9`, `seg_k_l15h7`,
`track_q`, `track_k`, `text_attention` and `depth`. Q and K jointly belong to one
tracking head, so these six matrices correspond to FIVE heads. There is no
concatenation/compression into a policy feature in this pipeline.

The semantic distribution excludes padding and retains separate word responses.
Summing all valid words gives one, so downstream selection must select a word,
phrase, or content-token response; it must not sum the entire instruction.

The depth output is unitless. Its future protocol transfer is explicitly recorded
as `unvalidated_protocol_transfer`; the earlier observed-depth evidence does not
validate this denoising state, camera distribution or inferred-region readout.

## Policy-side latent organization (design note, not yet the training contract)

Keep prompt localization and tracking separate. A prompt heatmap answers “where is
target n at this time?”; a tracking bundle answers “where do the C points inside
that target move?” The text is encoded once, while a prompt-conditioned heatmap is
read at each time position. At sparse anchors, that heatmap re-boxes the same target
and corrects the track; it does not create a new target identity.

```text
prompt_heat:     [B,T,N,32,32]
track_heat:      [B,T,N,C,32,32]
K_frame:         [B,T,32,32,Dk]
relative_depth:  [B,T,32,32,1]
anchor_flag:     [B,T]
```

`N` is the number of prompt-selected targets/phrases, padded to the batch maximum
with a `bundle_mask`. `C` is the fixed number of tracked points per target.
`track_heat[b,t,n,c]` is the 32x32 probability heatmap for point `c` of target `n`
at time `t`. `anchor_flag` is control metadata for track reset, not a policy
feature. K and relative depth remain per-frame policy inputs, but these fields are
not flattened into one feature vector.

## Teacher Timelines

Observed extraction retains the existing protocol:

```text
33 actual past RGB +15 copies of the first past frame -> control
main input = pure Gaussian noise at timestep1000
one frozen forward; full original task instruction
```

This produces 12 native positions. Warmup positions 0..3 are excluded from
direct observed losses. Anchors use native `[4,6,8,10]`, assigned to actual
RGB `[0,8,16,24]` by the existing convention.

Future extraction:

```text
49-frame GT VAE latent -> (1-sigma)*clean + sigma*Gaussian noise
control = past33 RGB, future16 pixels zero
mask = observed0, future1
40-step UniPC / flow_shift3; begin nearest sigma0.8
1 actual scheduler update to the original step-13 endpoint
then one CONDITIONAL forward at timestep499
```

`denoise_updates=1` counts actual updates; `denoise_reference_updates=13` fixes
the interval on the original schedule. The selected relative nodes are `[0,13]`;
the actual sigma goes from0.8019521832466125 to0.4996665418148041. This is a
sparse approximation, not the first original update or GT directly noised to
the endpoint. The final capture reads the actual noisy latent, not an `x0`
estimate. Every extraction creates a fresh UniPC history and retains unused
prefix/suffix nodes. The actual start/capture sigmas, schedule nodes and timestep
are written to `teacher_protocol.json`. All five future fields and teacher paths
come from that same final conditional forward.

`configs/vace_structured_step13_reference.json` retains all13 original updates.
The earlier protocol's checkpoints/caches are different supervision and must
not be presented as one-update targets. Strict checkpoint resume rejects a
teacher-protocol change; the optional cache reader rejects mismatched actual
update counts/reference endpoints. The observed branch and temporal mapping are unchanged.

The full future teacher timeline has 13 positions. Native8 includes the observed
boundary near RGB32 and is retained for boundary-to-future relations. Future
anchors are native `[8,10]`, assigned to RGB `[33,41]`. Native temporal receptive
fields span RGB frames: these mappings are a convention, not exact per-frame
features. No RGB-frame duplication is claimed as additional information.

QK targets are built within each extraction protocol. The code does not dot
observed-pure-noise Q with GT-noised-future K as if these two internal coordinate
systems were identical.

## Tracking Readout and Losses

Tracking primarily distills QK relations, not raw Q/K vector coordinates.
Sampled source queries cover the whole native grid. Adjacent and two-step
relations are supervised using normalized Q/K and temperature0.07.

Multi-anchor supervision uses independent 8-RGB-frame short paths from observed
anchors0/8/16/24 and future anchors0/8. The existing Q-to-K beam decoder uses
beam4, top32 candidates, motion/descriptor/region/acceleration terms and the
existing step/acceleration regularizer. It does not use optical flow or GT
object masks. Discrete decoding runs only on the frozen teacher. Student path
loss is differentiable soft-grid cross-entropy through its own QK scores.

Seed selection for all 40 full instructions is explicitly a new generic readout:
maximum response over non-special content tokens, expanded/filtered by the two
spatial K fields. It is NOT a claim to reproduce the earlier manually specified
`black bowl` phrase readout for every task. The per-anchor cap64 is not a fixed
point count per object. Path validity is temporal support, not calibrated
visibility or correspondence confidence.

Auxiliary terms are:

- Same-video mild RGB brightness/color consistency (one transform per clip).
  Tracking compares relations, not individual Q/K coordinates.
- Weak soft cycle consistency on pairs where the teacher's argmax matches are
  mutually consistent. This is a reliability gate, not GT visibility.
- Weak depth neighbor regularization where teacher K affinity is high. It does
  not impose a constant depth on an object and does not use temporal EMA.

Weights and pure-distillation ablations are configurable. These regularizers
have not been shown to improve closed-loop control. No fixed cluster count,
whitening, learned depth calibration or feature compression is introduced.

`train.head_weights` names all five heads: tracking defaults to weight2 and
the two partition heads, semantic head and depth head each default to weight1.
The joint objective sums these weighted observed+future losses, plus weighted
augmentation consistency. Each head's raw and weighted losses are logged
separately; equal weights do not claim equal numerical scales or gradient sizes.
Setting one weight to zero permits an ablation while retaining a connected
computation graph. The shared teacher installs all capture hooks in one forward:
the one-update future denoising sequence is run once per sample, not once per head.

## Data and Training

The earlier offline mode is retained for archived artifacts: eight fixed offset
windows per episode/camera, loss-only FP16 targets on TOS and a shared50 GB
local cache. See [VACE_TEACHER_OFFLINE.md](VACE_TEACHER_OFFLINE.md).
The current mainline extracts online; it does not start TOS caching.

The episode manifest covers all 1,693 local training trajectories, all 40 tasks,
273,465 stored frames and both cameras. The source is `datasets/libero`, LeRobot
v3, stored FPS10. The code preserves these timestamps; it does not silently
convert them to 20Hz actions. New simulator episodes are needed for independent
evaluation because the full declared training split is used.

For every epoch, each episode/camera contributes four random 49-frame windows.
Starts and noise seeds are deterministic functions of seed/epoch/sample id;
they change across epochs and never cross episode boundaries. Teacher features
are extracted online in microbatches of eight, with no saved feature cache or fixed
window list. Full future RGB is a separate DataLoader field consumed only by
the teacher. The student receives `batch['rgb']`, which contains past33 only.

Default recipe: per-rank student batch32, teacher microbatch8, accumulation1,
8 ranks, global batch256; backbone LR5e-5, decoder/head LR5e-4, warmup100,
cosine decay, gradient clip1,10000 optimizer updates.
`steps` counts OPTIMIZER updates. These are starting settings, not a convergence
or eight-GPU capacity claim. The resident batch2 teacher benchmark decreased
from72.59s (13 updates) to17.99s (one update), about4.04x, including VAE and
CPU beam decoding. This excludes student training and is a single shared-A800
measurement; see `../runs/vace_sparse_teacher_speed_20261010/summary.json`.
New runs write to `../runs/vace_structured_online_step1/<task>/`.

```bash
bash /increase_kairos_vepfs/increase/liwenhao/agent/2337888765/heft/latent_policy/scripts/launch_vace_online_structured.sh
```

Existing processes retain their loaded settings; this change does not launch or
restart sustained training.

FSDP2 shards only the student, including the five head modules. Teacher/VAE are frozen local replicas per rank;
UMT5 is unloaded after building the cache. DDP is an optional config alternative.
Checkpoints are collective full student+optimizer saves with per-rank RNG,
epoch and DataLoader cursor. Resume checks architecture, teacher settings,
data/text hashes, batch/optimizer/loss settings and world size. Save interval100;
the final update is also saved. Teacher weights are not stored in checkpoints.
Logs distinguish `distillation_total` from the actual optimized `total`, which
also includes weighted augmentation consistency. Loading-stage events precede
the first optimizer update.

## Commands

Run from `heft`:

```bash
source latent_policy/scripts/env.sh
"$LATENT_POLICY_PYTHON" latent_policy/scripts/build_vace_latent_manifest.py \
  --root ../datasets/libero --output ../runs/vace_latent_data/episodes.jsonl

CUDA_VISIBLE_DEVICES=0 "$LATENT_POLICY_PYTHON" latent_policy/scripts/build_vace_text_cache.py \
  --manifest ../runs/vace_latent_data/episodes.jsonl \
  --text-model ../models/Wan2.1-T2V-1.3B-Diffusers \
  --output ../runs/vace_latent_data/text.pt

CUDA_VISIBLE_DEVICES=0 "$LATENT_POLICY_PYTHON" latent_policy/scripts/smoke_vace_structured.py \
  --task shared --output ../runs/vace_structured_step1_smoke

bash latent_policy/scripts/launch_vace_online_structured.sh
# Independent ablations: seg_l22h9, seg_l15h7, tracking, semantic, depth
# Single GPU: NPROC_PER_NODE=1 bash .../launch_vace_online_structured.sh
```

The last command explicitly starts training; code preparation does not start it.
Use `--resume /absolute/checkpoint.pt` and a larger `--steps` to extend one run.
For a pure distillation comparison, set auxiliary loss weights to zero in a
separate config/output; do not change them while resuming an existing experiment.

Tests: `python -m pytest -q latent_policy/structured_tests`. Real smoke results
are written separately; unit-test success alone is not an eight-GPU benchmark.

The bounded training acceptance recipe performs exactly one optimizer update,
including online teacher extraction, augmentation consistency and a checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 "$LATENT_POLICY_PYTHON" latent_policy/scripts/train_vace_structured.py \
  --config latent_policy/configs/vace_structured_shared_one_update.json
```

Student-only inference accepts a video path but decodes only 33 frame indices
starting at `--start`. Exported NPZ files include native observed/future matrices
and timeline validity/anchor metadata. A shared checkpoint contains all five
heads and the common encoder. An independent checkpoint contains its selected
head. Independent and shared architectures cannot be resumed interchangeably.

```bash
CUDA_VISIBLE_DEVICES=0 "$LATENT_POLICY_PYTHON" latent_policy/scripts/infer_vace_structured.py \
  --checkpoint ../runs/vace_structured_smoke/step1_shared_one_update/shared/step_0000001.pt \
  --video ../datasets/libero/videos/observation.images.image/chunk-000/file-000.mp4 \
  --start 142 --fps 10 \
  --instruction "put the white mug on the left plate and put the yellow and white mug on the right plate" \
  --output ../runs/vace_structured_smoke/student_shared.npz
```

## One-Update Mainline Verification (2026-10-10)

- 28 CPU checks passed for schedules, student boundaries/losses/checkpoints and
  offline serialization. One/two/four updates share the original step-13 endpoint;
  the13-update schedule produces bit-identical solver outputs to the original.
  Parallel and sequential CFG both capture after the actual sparse update.
- The cache-reader check rejects archived13-update supervision under the new
  one-update recipe before RGB decoding. Its six offline checks were rerun after
  adding this guard and passed.
- A real single-A800 shared-head smoke completed teacher extraction, all five
  observed/future losses and V-JEPA2 backward. The protocol records `[0,13]`,
  actual update count1, capture sigma0.4996665418148041 and timestep499.
  Backbone/text projection/future-query gradient norms were16.98/29.99/0.424,
  all finite and nonzero. Evidence:
  `../runs/vace_structured_step1_smoke_20261010/shared.json`.
  This batch1 smoke has no optimizer update and is not batch32/eight-GPU capacity
  or representation-quality validation. The process exited after the check.

## Historical Independent-Head Verification (13 updates)

- All 10 CPU checks pass, including every head's forward/backward, text influence
  on future output, rejection of future RGB input, and exact optimizer-checkpoint
  restoration.
- A real tracking window passes frozen-teacher extraction and ViT-L backward.
  Actual future sigma is 0.801952, with capture timestep499 after 13 updates.
  Backbone, text projection and future query all have finite, nonzero gradients.
- A two-rank small-model FSDP2 check passes backward and collective
  save/restoration with zero output discrepancy. This checks the sharding
  mechanism, not full-model eight-GPU capacity.
- The real single-GPU training entrypoint completes one optimizer update with
  teacher and student resident together, augmentation backward enabled and a
  full student/optimizer checkpoint written. Per-rank batch is one; the timed
  update takes 62.95s, excluding initialization and checkpoint IO. This is an
  execution check, not a batch2 memory-capacity or convergence result.

Machine-readable smoke evidence is stored under `../runs/vace_structured_smoke`.
Housekeeping on 2026-10-11 removed its disposable one-update checkpoints, dummy
FSDP checkpoints and NPZ exports; JSON protocols, metrics and verification reports
remain at their original paths. Other historical short-trial reports were
consolidated under `../runs/_test_records/`. Regenerate a smoke checkpoint before
using it for an inference command; these records are not a checkpoint backup.
The standalone student's measured 4.03GiB smoke peak excludes optimizer state
and teacher coexistence; it must not be used as a total training memory estimate.
These checks verify execution, not representation quality or policy success.

## Historical Shared Five-Head Verification (13 updates)

- All 13 CPU checks pass, including encoding the backbone once, nonzero gradients
  in all five heads, configurable joint loss weights, and restoration of every
  shared-checkpoint output.
- A two-rank small-model FSDP2 run completes backward, optimizer update and
  collective save/restore. All future head outputs restore with maximum error0.
  Evidence: `../runs/vace_structured_smoke/fsdp_shared/fsdp_check.json`.
- A real single-GPU shared run completes one optimizer update with online teacher,
  all five observed/future losses, augmentation backward and checkpoint saving.
  Batch1 peak PyTorch allocation is 13.76GiB; the timed update is 49.56s,
  excluding initialization and checkpoint IO. This is not full eight-GPU or
  batch2 capacity verification. Evidence:
  `../runs/vace_structured_smoke/shared_one_update/shared/metrics.jsonl`.
- Student-only inference reloads this shared checkpoint and successfully exports
  all six matrices in each branch plus timeline/anchor metadata, using only the
  33 past frame indices142..174 and the task instruction. It does not instantiate
  the teacher. Evidence: `../runs/vace_structured_smoke/student_shared.npz` and
  its JSON sidecar. The one-update outputs are execution evidence, not trained
  representation-quality results.
