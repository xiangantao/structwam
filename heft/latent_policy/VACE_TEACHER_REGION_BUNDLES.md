# Teacher Scene Fields and Named Region Bundles

This readout experiment uses the current VACE teacher; it does not load or train
the student. Relative depth is excluded from this visualization.

Two scene-wide normalized K fields (L22H9 and L15H7) and the L15H4 post-RoPE Q/K
are shared by every bundle. They cover all 32x32 cells, including background,
robot and objects absent from the instruction. The teacher is conditioned on the
full instruction; the public readout does not filter cells by text relevance.
K vector colors use a single label-free PCA3 and fixed limits per head across
both branches and all cases. This projection is for display only; all128 K
coordinates are retained. It is not instance segmentation or a preset cluster count.

Every public anchor seeds all1024 cells. The existing QK+beam decoder is used
with a constant region gate. The public video draws every fourth row/column
for readability; the arrays retain every point.

Named bundles read all occurrences of explicitly supplied noun phrases from
the same full-instruction embedding and same teacher forward. This diagnostic
does not add a noun parser. Each bundle contains independent word-response
heat, prompt/K region heat, dense point paths, temporal support and point
occupancy heat. The full-instruction overview uses maximum content-token
response; it never sums the complete normalized word distribution.
Phrase colors identify readout slots, not ground-truth objects. Overlap is
retained; there is no forced disjoint assignment or GT mask in decoding.

Observed extraction:33 real frames +15 copies of frame0, pure-noise main input.
Future extraction:full49-frame GT latent + sigma0.801952 noise, control past33,
one update to the original step13 endpoint sigma0.499667, then conditional capture.
The source video backgrounds are actual recorded RGB including future GT;
they are not generated RGB or a one-update latent decode. Anchors and temporal
interpolation follow the existing convention.

Each case saves `teacher_fields.pt` and `region_bundles.npz`. The latter contains:

| Field | Shape per branch |
| --- | --- |
| `prompt_heat`, `region_heat`, `trajectory_heat` | `[S,T,32,32]` |
| `trajectory_heat_by_anchor` | `[S,A,T,32,32]` |
| `paths`, `raw_paths` | `[S,A,P,T,2]` |
| `path_valid` | `[S,A,P,T]` |
| `seeds` | `[S,A,P]` |
| `public_paths`, `public_raw_paths` | `[A,1024,T,2]` |
| `public_path_valid` | `[A,1024,T]` |

`S` is one full-instruction overview plus the supplied phrases. `T/A` are12/4
for observed and13/2 for future. `P` is the per-anchor cap64, not a point count
per physical object. Short paths retain individual point and anchor identities;
their bilinear occupancy is an additional view, not a replacement for correspondence.
Validity denotes temporal support, not predicted occlusion confidence.

Outputs separate public K, public QK tracks, raw phrase responses, prompt/K
regions and bundle occupancy. Phrase-specific videos show response, region,
and region with tracks side by side. `diagnostics.json` reports phrase seed
overlap without interpreting it as semantic accuracy.

```bash
source /increase_kairos_vepfs/increase/liwenhao/agent/2337888765/heft/latent_policy/scripts/env.sh
CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=1 "$LATENT_POLICY_PYTHON" \
  /increase_kairos_vepfs/increase/liwenhao/agent/2337888765/heft/latent_policy/scripts/render_vace_teacher_region_bundles.py \
  --output /increase_kairos_vepfs/increase/liwenhao/agent/2337888765/runs/vace_teacher_region_bundles_20261010_v1
```

The default examples are episodes39 and40, both cameras, with three and two noun
phrases respectively. `--render-only` reuses these bounded diagnostic fields
on CPU. This does not restart the abandoned full-dataset/TOS cache.
