# VACE Offline Teacher Dataset for Structured Distillation

Archived plan: the user discontinued TOS caching. The current mainline uses
online one-update extraction; see `VACE_STRUCTURED_DISTILLATION.md`.
The locations, costs and 13-update protocol below describe the historical run,
not one-update targets. New extraction uses a distinct provenance, including
the sparse scheduler source; do not append to the old run.

This mode materializes the existing five-head teacher contract. Visual targets
are extracted offline; student training reads TOS through a bounded local cache,
without loading VACE or decoding future RGB.

## Locations

- TOS: `tos://kairos-vla-datas/research/2337888765/vace_teacher/libero_8offset_lossfp16_20261009_v1/`
- Cache: `/increase_kairos_vepfs/increase/liwenhao/agent/2337888765/cache/vace_teacher_libero8`
- Logs/manifests: `/increase_kairos_vepfs/increase/liwenhao/agent/2337888765/runs/vace_teacher_offline_libero8_20261009`

This environment can access `kairos-vla-datas`; legacy `vla-datas` returns403.
TOS requests use the provisioned config explicitly and bypass the proxy. No
credentials are changed or serialized. Use `--tos-config` on another machine.

## Sampling and Protocol

All1,693 episodes,40 tasks and two cameras contribute eight distinct49-frame
windows each:27,088 artifacts. Starts span the episode and cover all eight
phases modulo8; paired cameras share starts. This varies physical frames'
relative placement within windows, without enumerating every possible context.
Starts/noise seeds are fixed. Training shuffles these alternatives across epochs
and can augment RGB; it does not regenerate labels each epoch.

Each file records episode, camera, instruction, absolute frame indices, FPS10,
start, seed, protocol id and actual extraction scheduler state. Provenance hashes
cover the episode manifest, text cache, teacher config and readout sources.
Checkpoint names/sizes/mtimes are recorded; these are not full content hashes.
Changing the immutable extraction recipe requires a new run.

Observed: past33 plus15 copies of the first frame, main pure noise, one frozen
forward. Future:49 GT latents at nearest sigma0.8, control past33,13 CFG UniPC
updates then conditional capture. The recipe runs28 Transformer calls per
window:1 observed +26 future-denoising +1 final capture. All five heads are
captured together, without repeating denoising for each head.

| Branch | Original native length | Saved native indices | Saved length |
| --- | --- | --- | --- |
| Observed | 12 | 4..11 | 8 |
| Future | 13 | 8..12 | 5 |

Future native8 preserves the observed boundary for temporal relation losses.
Save two partition K fields, tracking Q/K,512-word prompt attention and raw
relative depth as FP16. Keep32x32 cells, int64 seeds, FP32 beam paths and boolean
validity. No spatial pooling/PCA/cluster count is introduced. The reader restores
12/13 time slots with zeros at unused positions; existing loss masks exclude
them. FP16 introduces small rounding differences from online FP32.
Future depth remains `unvalidated_protocol_transfer`.

Approximately27.3 MB per file,0.74 TB decimal overall, plus small manifests and
the329 MiB text cache. RGB and intermediate denoising states are excluded.

## Queue and Cache

Four GPU processes claim different windows; two separate processes upload.
Defaults:GPU0/2/4/6. Existing GPU jobs are not terminated.
The global cache limit is50,000,000,000 bytes, about46.6 GiB. Writers reserve
capacity in SQLite before writing, leaving room for index/allocation overhead.
Unuploaded files and pinned readers cannot be evicted. Verified remote copies
are reclaimed by LRU when needed; extraction waits if no space can be reclaimed.
Uploads verify CRC64 and remote byte count, record SHA256; downloads verify SHA256.
Upload failures retain files and retry. Worker crashes recover owned entries.
Restarting the same run skips remote artifacts even after local eviction.

`windows.jsonl` is planned work. `uploaded.jsonl` contains only verified remote
artifacts and is refreshed by the coordinator. Training loads a fixed verified
snapshot. A partial snapshot is only a small-data check, not the full dataset.

## Commands

From `heft`, prepare and start a detached extraction coordinator:

```bash
bash latent_policy/scripts/launch_vace_teacher_cache.sh --gpus 0,2,4,6 --uploaders 2
```

Inspect, stop owned workers preserving resumable state, or restart after exit:

```bash
source latent_policy/scripts/env.sh
"$LATENT_POLICY_PYTHON" latent_policy/scripts/cache_vace_teacher.py --role status
"$LATENT_POLICY_PYTHON" latent_policy/scripts/cache_vace_teacher.py --role stop
"$LATENT_POLICY_PYTHON" latent_policy/scripts/cache_vace_teacher.py --role start
```

The verified snapshot can later feed the existing8-GPU FSDP2 student:

```bash
bash latent_policy/scripts/launch_vace_structured.sh shared \
  --offline-manifest ../runs/vace_teacher_offline_libero8_20261009/uploaded.jsonl \
  --teacher-cache-dir ../cache/vace_teacher_libero8 \
  --output ../runs/vace_structured_offline_libero8
```

Extraction does not start student training. The online default remains available
when `--offline-manifest` is absent. Tests cover loss/path retention, offset
diversity, bounds, pins, concurrent reservations, recovery, checksums and
past-only decoding. Real extraction/upload evidence is in the run logs.
