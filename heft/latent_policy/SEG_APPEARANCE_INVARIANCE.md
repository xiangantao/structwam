# Optional segmentation appearance consistency

The segmentation heads still match the teacher K vectors with pointwise cosine
distillation. An opt-in auxiliary objective encourages the student K vectors to
stay consistent when past RGB has mild appearance changes. It uses no new object
labels, masks, cluster counts, spatial smoothing, or teacher extraction.

For each seg head and valid token, the auxiliary error is:

```text
1 - cosine(student_K(augmented_past_RGB, instruction),
           stopgrad(student_K(clean_past_RGB, instruction)))
```

Observed and future outputs are both constrained. Each branch is averaged over
its existing valid tokens, then combined as
`(observed + future_weight * future) / (1 + future_weight)`.
Shared mode sums the two seg objectives with the existing head weights. The
auxiliary multiplier is `train.consistency_weight` (default `0.05`). The student
still receives exactly 33 past frames and the unchanged instruction; future RGB
is not a student input.

## Enable and compare

Add `--seg-invariance alternating` to the existing training command, using a
separate output directory for the experiment. The launcher forwards this flag:

```bash
bash scripts/launch_vace_structured.sh shared \
  --seg-invariance alternating \
  --output /increase_kairos_vepfs/increase/liwenhao/agent/2337888765/runs/vace_student_seg_appearance
```

This example assumes the existing config, data and text cache are ready. Keep
the same online/offline input options as the baseline training command.

| Mode | Seg consistency augmentation |
| --- | --- |
| `none` | Existing weak consistency; default baseline |
| `color` | Clip-coherent brightness, contrast, saturation and partial grayscale |
| `texture` | Mild Gaussian blur and downsample/restore, blended with original RGB |
| `alternating` | Alternate color and texture on successive microbatches |

Color parameters are sampled once per clip, shared by all frames. Brightness and
contrast are in `[0.8, 1.2]`; saturation is in `[0.6, 1.4]`. Partial grayscale
occurs with probability `0.1`, blending at most `0.35`. Texture mode uses a 3x3
Gaussian with sigma `[0.4, 0.8]`, resize ratio `[0.85, 1.0]`, and blend weight
`[0.25, 0.6]`. There is no geometric or temporal augmentation, so token positions
retain their alignment.

The stronger augmentation contributes loss only through seg outputs. In shared
mode the backbone also receives these gradients, so other heads can change
indirectly. Non-seg heads retain the original weak consistency. With all five
heads active, enabling this option adds one student forward/backward per
microbatch to the baseline's clean and weak-augmentation passes. Teacher target
files and extraction protocols are unchanged. Independent seg mode replaces the
existing weak pass instead of adding a third pass.

## Resume and validation

Enabled runs record the mode, augmentation source hash and recipe version in
checkpoint provenance and `seg_invariance.json`. Resume with the same mode and
code; switching from a baseline checkpoint to an enabled mode is rejected by the
existing provenance check. Mode `none` adds no provenance keys.

CPU tests cover clip coherence, texture attenuation, valid-time masking,
stop-gradient targets, both seg heads, shared-backbone gradients, and independent
seg mode:

```bash
.venv/bin/python -m pytest -q \
  structured_tests/test_seg_invariance.py structured_tests/test_pipeline.py
```

These checks verify implementation behavior, not better object separation. Use
the same held-out episodes/windows to compare baseline, color-only, texture-only,
and alternating runs. Check feature consistency under appearance perturbations
alongside within-object compactness and between-object separation; low auxiliary
loss alone does not establish better partitions. No training or quality
evaluation is started by adding this option.
