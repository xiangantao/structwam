# VACE Attention Motion Readout: Observed Structure, Prompt/K Selection, QK Tracks, and GT-Noised Short-Future Diagnosis

Status: one-update mainline, 2026-10-10. The 2026-10-08 case and artifacts below
remain historical references.

This is the normative English specification for the current VACE experiment. It
documents how the VACE video generator is used as a representation reader:

```text
observed video + task prompt
        -> prompt localization and K spatial organization
        -> QK correspondence decoding
        -> observed object/gripper tracks

observed condition + GT future latent corrupted with sigma=0.8
        -> short denoising diagnostic
        -> future motion-structure readout
```

The multi-anchor decoder is only one part of this design. The scientific purpose is
to separate three questions that were previously mixed together:

1. Does the model expose useful spatial structure for the current observation?
2. Can prompt attention select the task-relevant part of that structure while K
   preserves spatial grouping?
3. When a future latent is deliberately made recoverable but noisy, does an
   intermediate denoising representation expose motion structure?

The Chinese review version is
[`VACE_ATTENTION_MOTION_READOUT_PIPELINE_ZH.md`](VACE_ATTENTION_MOTION_READOUT_PIPELINE_ZH.md).

## Current mainline: one update to the original step-13 endpoint

The observed protocol remains 33 real past frames plus 15 copies of frame0 for
prompt/K warmup. The student takes only the 33 real past frames and the instruction.
The future teacher still uses the full 33+16 GT window for its noisy main latent,
with only the past33 visible in the control condition and the full task prompt.
The purpose is recovering motion structure; texture quality is not the criterion.

```text
GT VAE latent + noise at sigma0.8019521832466125, timestep801
    -> ONE CFG UniPC update across the reference interval [0,13]
    -> actual noisy latent at sigma0.4996665418148041, timestep499
    -> one CONDITIONAL forward to capture all future fields
    -> prompt heat / K regions / seeds / QK / multi-anchor beam tracks
```

`teacher.denoise_updates=1` counts actual updates.
`teacher.denoise_reference_updates=13` fixes the endpoint on the original
40-step UniPC schedule, flow_shift3, CFG5. This is not the first update of the
old schedule and not GT directly corrupted to sigma0.50. All future fields are
recomputed from the same endpoint state. Scheduler history resets per extraction;
unused prefix/suffix nodes remain to preserve UniPC solver behavior. The
protocol records the actual start/capture sigma, timestep and selected nodes.
Tracking heads, 8-frame anchors, beam constraints and temporal mapping are unchanged.

The 13-update reference is retained in
`configs/vace_structured_step13_reference.json`. One update approximates the
longer trajectory; equal endpoint sigma does not imply equal latent or features.
The fixed-query comparison on episode39 agentview, using visible displaced GT
points and excluding anchors/static objects, gave:

| Updates | Moving-object EPE, source256px | Gripper EPE, source256px |
| ---: | ---: | ---: |
| 1 | 20.43px | 23.70px |
| 13 | 22.06px | 24.50px |

This is evidence for this case, not a cross-episode quality ranking. Results are
in `../runs/vace_sparse_future_lowsteps_ep39_20261010/summary.json`.
The preview RGB is shared from the old fully denoised reference; only the future
attention fields, regions, seeds and tracks are regenerated for each schedule.

On a resident A800 teacher, batch2 with parallel CFG batch4, the earlier synchronized
measurement gave 17.99s for the full one-update teacher versus 72.59s for 13
updates (about4.04x). Future denoising plus capture took5.13s versus60.00s. This
includes all five readouts and CPU beam decoding, excludes initialization/student
training, and is one measurement on a shared GPU. It is not an eight-GPU throughput
claim. Source: `../runs/vace_sparse_teacher_speed_20261010/summary.json`.

The online launcher reads the new default recipe without additional arguments:

```bash
bash /increase_kairos_vepfs/increase/liwenhao/agent/2337888765/heft/latent_policy/scripts/launch_vace_online_structured.sh
```

New runs go to `../runs/vace_structured_online_step1/<task>/`. Existing processes
keep their loaded settings. A 13-update checkpoint is a different teacher
protocol and cannot be resumed under the one-update recipe by the strict resume
check. Historical artifacts and their stored protocols are not relabeled.

## What this experiment is and is not

The observed branch is a representation and correspondence diagnostic. It reads
attention-derived spatial fields and renders tracks; it is not a closed-loop policy,
an object detector with GT masks, or proof that every track has the correct object
identity.

The future branch is a privileged teacher diagnostic. It tests whether motion can be
read from a denoised, noisy future latent while the control condition contains only
the observed prefix. It is **not** a causal future-prediction benchmark because the
main latent initialization contains the future GT video. A later causal predictor
or distillation student must receive past observations only.

## Frozen case and artifacts

| Item | Value |
| --- | --- |
| Dataset | LIBERO |
| Suite / episode | `libero_spatial`, episode `39` |
| Camera | `agentview` |
| Source video | `/increase_kairos_vepfs/increase/liwenhao/agent/2337888765/runs/vace_tracking_all_suites_20261004/videos/episode_00039/agentview.mp4` |
| Actual observed source window | native frames `48..80` inclusive, 33 frames |
| Python slice used | `48:81` |
| Task prompt | `pick up the black bowl on the wooden cabinet and place it on the plate` |
| Prompt phrase used for localization | `black bowl` |
| VACE model | `/increase_kairos_vepfs/increase/liwenhao/agent/2337888765/models/Wan2.1-VACE-1.3B` |
| Text model | `/increase_kairos_vepfs/increase/liwenhao/agent/2337888765/models/Wan2.1-T2V-1.3B-Diffusers` |
| Final artifact directory | `/increase_kairos_vepfs/increase/liwenhao/agent/2337888765/runs/vace_qk_multi_anchor8_20261008_ep39` |

The final H.264 previews are 512x512 at 10 FPS:

```text
observed_multi_anchor8_h264.mp4              33 frames
future_gt_sigma080_multi_anchor8_h264.mp4    16 frames
observed_then_future_multi_anchor8_h264.mp4  49 frames
```

The 10 FPS value is the recorded/rendered video rate. It must not be silently
interpreted as the simulator's 20 Hz action rate.

## The observed input: 33 real frames, 48 model slots

There are two different frame counts, and both must be recorded:

| Meaning | Count | Contents |
| --- | ---: | --- |
| Real observed RGB frames | 33 | source frames `48..80`, local indices `0..32` |
| VACE RGB slots in the observed readout | 48 | 15 warm-up copies plus the 33 real frames |
| Rendered observed output | 33 | only the real observed timeline is shown |

The observed VACE input is exactly:

```text
[actual frame 0] repeated 15 times
followed by
[actual frame 0, actual frame 1, ..., actual frame 32]
```

Thus the model sees 48 temporal slots, but only 33 of them are real observations.
The first 15 slots are copies of the first real frame. They exist to stabilize the
prompt/K prototype before the real observation timeline is read. They are not extra
data, are not displayed, and must not be counted as 15 additional observations.

With the VAE's approximately 4x temporal compression, the 48-slot observed input
produces 12 latent time positions. The actual frame-0 prototype is at latent index
4. The saved observed anchors are:

```text
native observed frames:  0,  8, 16, 24
latent anchor indices:    4,  6,  8, 10
```

The actual observed frame 32 is the end of the observed timeline and the boundary
before the future diagnostic. It is not stored as an additional observed anchor in
the final NPZ. Latent coordinates are linearly interpolated to the 33 raw frame
times for rendering; one latent coordinate is not held for four RGB frames.

## Why the future branch uses GT plus sigma=0.8 noise

The choice of GT plus noise was made to answer a narrower question than “can VACE
predict the future from the past?” The earlier pure-noise continuation mixed several
failure modes together:

- the generator had to infer the future trajectory;
- it had to synthesize a plausible image;
- it had to maintain temporal identity;
- and the readout had to decide whether a noisy intermediate feature contained
  motion.

If the pure-noise video looked wrong, that result could not distinguish “the feature
has no motion information” from “the untrained/untuned generator failed to produce
the future.” Conversely, giving the main branch a clean future would make the
readout close to trivial restoration and would not test a useful denoising state.

The GT-assisted corruption isolates the representation question:

```text
latent_init = (1 - sigma) * VAE(full 49-frame GT window)
              + sigma * Gaussian noise
```

The observed prefix remains available through the control condition, while the
future part of the main latent is corrupted. A noise scan showed the practical
trade-off for the 16-frame diagnostic:

- lower noise leaves too much of the future intact and mostly measures restoration;
- pure noise or very high noise makes the continuation failure dominate the
  readout;
- sigma=0.8 leaves enough corruption to require denoising while still allowing a
  short motion-structure signal to emerge before all texture is fully refined.

At the selected scheduler point, the requested sigma is `0.8` and the actual flow
sigma is `0.8019521832466125`. The run begins at schedule index 17 of a 40-step
UniPC schedule, leaving 23 denoising updates. The earlier step-by-step inspection
showed that motion was already visually readable around the middle of denoising,
before the last steps finished texture cleanup. That is why this branch is useful as
a motion-structure teacher diagnostic.

This does **not** make sigma=0.8 a universal optimum, nor does it make the branch
causal. The full GT future participates in the main initialization by design.

## Future input timeline and causal boundary

The future diagnostic uses a 49-frame window:

```text
actual observed prefix: 33 frames, source indices 48..80
diagnostic future:      16 frames, source indices 81..96
total GT window:        49 frames
```

The control condition contains the observed prefix only:

```text
observed control pixels: real RGB
future control pixels:   zero
observed mask:           0
future mask:             1
```

The main branch receives the full 49-frame GT latent mixed with Gaussian noise at
sigma=0.8. Therefore the branch is “observed condition + future-informed noisy main
latent,” not “observed condition -> causal prediction.” The final future video shows
only the 16-frame future segment, while the QK readout keeps the 13-position latent
timeline of the full 49-frame window.

## Attention paths and their roles

The final readout separates the roles of prompt, K, and QK rather than treating a
single QK matrix as a generic feature map:

| Readout | Path | Role |
| --- | --- | --- |
| Prompt localization | main `attn2`, `L22H4` | selects positions associated with the phrase `black bowl` |
| Spatial K organization | K fields `L22H9` and `L15H7` | keeps prompt seeds inside a coherent spatial region |
| Track correspondence | main self-attention `attn1`, `L15H4` | reads where a selected grid cell moves next |

The forward pass is conditioned by the VACE control input. The saved Q/K readout,
however, is the selected model-block attention path above; it is not a GT segmentation
mask and it is not a separately trained object encoder.

## Prompt-to-K region proposal

The phrase tokens for `black bowl` are localized by summing their `attn2` attention
probabilities at `L22H4`, producing a 32x32 spatial prompt field. The K fields at
`L22H9` and `L15H7` are L2-normalized per grid cell. Prompt scores propose seeds;
K-vector affinity expands and filters the seeds toward spatially coherent cells.

Up to 64 grid cells are retained per anchor. The procedure does not use simulator
object masks, hand-selected cluster centers, optical flow, or a fixed number of
objects. Static candidate cells are not discarded merely because their object does
not move; they remain eligible if they are selected by the prompt/K region.

Prompt color rendering uses one observed global range, approximately p1-p99 over
all observed latent times. It does not stretch each frame independently. Independent
frame stretching was explicitly excluded because it amplified tiny early-frame
feature changes into false color motion.

## QK correspondence decoder

The selected tracking head is `L15H4`, with the frozen direction:

```text
q_t -> k_(t+1)
```

For every selected anchor cell, the decoder scores candidate cells in the next
32x32 grid using QK similarity. It combines that score with local descriptor
continuity, the time-varying K-region gate, velocity, and acceleration. Beam search
keeps several candidate paths before choosing the best path, which reduces isolated
argmax jumps.

| Setting | Value |
| --- | ---: |
| Beam size | 4 |
| Candidate top-k | 32 |
| Candidate max step | 4.0 grid cells |
| Motion penalty weight | 0.15 |
| Local descriptor weight | 0.25 |
| Region gate weight | 0.08 |
| Acceleration penalty weight | 0.05 |
| Descriptor EMA | 0.1 |
| Postprocess max step | 3.0 grid cells |
| Postprocess max acceleration | 1.5 grid cells |
| Regularization passes | 2 |
| Optical flow | disabled |

The postprocess only removes isolated jumps and excessive acceleration. It does not
replace QK decoding with optical flow and does not use GT coordinates.

## Why multiple anchors are used

The multiple-anchor design is a stability measure inside the larger readout pipeline,
not its scientific objective. A single anchor can drift over a long VAE-compressed
sequence. The observed branch therefore starts independent short tracks at native
frames `0, 8, 16, 24`, each for roughly eight raw frames. Their overlaps are rendered
together. This makes the point field denser and limits accumulated drift.

The future branch starts local anchors at future frames `0` and `8`; in the combined
49-frame timeline these are global positions `33` and `41`. Its stored latent anchors
are `[8, 10]` in the full generated-video timeline.

## Temporal rendering

Q/K tensors exist at latent positions, not independently at every raw RGB frame.
Adjacent latent coordinates and decoded trajectory points are linearly interpolated
onto the raw 10 FPS timeline for visualization. This is only temporal rendering; it
does not add information and does not use optical flow.

Observed and future videos are rendered separately and concatenated afterward. Any
purple/green colors in the combined preview are display conventions, not learned
channels.

## Saved outputs

`multi_anchor_readout.npz` contains:

```text
observed_heat             [12, 32, 32]
observed_region_scores    [12, 1024]
observed_anchor_latents   [4] = [4, 6, 8, 10]
observed_seeds            [4, 64] object array of selected grid cells
future_heat               [13, 32, 32]
future_region_scores      [13, 1024]
future_anchor_latents     [2] = [8, 10]
future_seeds              [2, 64] object array of selected grid cells
```

`heat` is the prompt-derived semantic field. `region_scores` is the K-derived gate
used by tracking. Neither is a GT segmentation mask. Seeds are 32x32 token-grid
cells, not pixel-accurate surface-point labels.

The final directory also contains:

```text
observed_multi_anchor8_h264.mp4
future_gt_sigma080_multi_anchor8_h264.mp4
observed_then_future_multi_anchor8_h264.mp4
multi_anchor_readout.npz
protocol.json
```

## Machine-readable contract

The final `protocol.json` records the non-negotiable choices:

```json
{
  "observed_warmup": "15 copies of actual frame0",
  "observed_anchor_native_frames": [0, 8, 16, 24],
  "future_anchor_native_frames": [33, 41],
  "anchor_stride_frames": 8,
  "requested_sigma": 0.8,
  "actual_sigma": 0.8019521832466125,
  "qk_layer": "L15H4",
  "direction": "q_t -> k_(t+1)",
  "optical_flow_used": false
}
```

## Read-only inspection

```bash
cd /increase_kairos_vepfs/increase/liwenhao/agent/2337888765
cat runs/vace_qk_multi_anchor8_20261008_ep39/protocol.json
python - <<'PY'
import numpy as np
p = "runs/vace_qk_multi_anchor8_20261008_ep39/multi_anchor_readout.npz"
z = np.load(p, allow_pickle=True)
for name in z.files:
    print(name, z[name].shape, z[name].dtype)
PY
```

The existing short-horizon script can regenerate the GT-assisted denoising
diagnostic. The exact final multi-anchor composite was produced by a one-off driver
that was not checked into the repository. The final artifact directory and its
`protocol.json` are therefore the source of truth for the 2026-10-08 result; the
current repository must not be described as providing a one-command bitwise replay.

## Excluded historical variants

These are not the current protocol:

- pure-noise future continuation as the main future result;
- QK readout from the pure-noise future run;
- Lucas-Kanade or another optical-flow future compositor;
- a single first-frame anchor;
- a real-frame-0 anchor without the 15-copy warm-up;
- independently stretched prompt heatmaps per frame;
- GT masks, hand-chosen cluster centers, or fixed object-count assumptions.

## Interpretation limits

The observed branch tests whether attention-derived fields can expose a coherent
current spatial/temporal structure. It does not establish closed-loop control or
causal object dynamics. The future branch has future GT in its main initialization.
Any distillation, action-conditioned prediction, or policy experiment must use a
separate past-only input protocol and report that causal boundary explicitly.
