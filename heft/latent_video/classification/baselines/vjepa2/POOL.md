# Training Pool Entry

Use one worker instance with eight A800 80GB GPUs, the same CPU/memory allocation
as the previous successful Wan job, and a root container user. Mount vePFS read/write
at `/increase_kairos_vepfs/increase`. Mount the complete workspace: the Python base
runtime in `cache/python`, V-JEPA2 source, models and dataset are adjacent to `heft`.

Use this plain image reference in the platform image field:

```text
iag-ad-01-cn-shanghai.cr.volces.com/wordmodel/base_env_ubuntu_2204_cu126_py310:v250910_v1
```

Do not include `http://`, brackets or Markdown link syntax. The earlier malformed
reference prevented the container from starting, before any training script ran.

Use this command once per node, without an additional outer `torchrun`:

```bash
bash /increase_kairos_vepfs/increase/liwenhao/agent/2337888765/heft/latent_video/classification/baselines/vjepa2/pool_launch.sh
```

The entry explicitly chooses eight ranks and two extraction lanes per GPU, each
handling 32 clips. The classifier batch is 32 videos per rank (256 globally), with
two segments per video. It uses the existing 20-epoch configuration and disables
in-loop validation. W&B stays offline. No downloads, dependency installation or
cloud submission happen in this script. Python, weights and videos are read
directly from the shared filesystem; the entry does not copy them to local storage.

## Startup Sequence

1. Save preparation stdout/stderr to shared `launch_logs/pool_*.log` immediately.
2. Check the mounted shared Python, model access and existing output checkpoint.
3. Use the shared venv and pretrained weights directly.
4. Import the real dependencies; test checkpoint rename/hardlink and local Unix sockets.
5. Check eight visible GPUs and BF16 matrix execution on each, then launch DDP.

Only multiprocessing temporary files and DataLoader shared-memory buffers use
node-local memory. The environment and weights remain on shared storage.

The original image has Python 3.10/CUDA 12.6 in its name, but the runner uses the
prepared Python 3.12.14 environment with PyTorch 2.11.0+cu128. Node checks verify
the runtime actually used and CUDA execution rather than relying on the image tag.
The container's driver/library configuration is preserved. The baseline is eager
and does not compile Transformer blocks. The historical `ptxas` execute-bit
problem is checked and reported separately; this baseline does not require it.

## Check And Resume

These checks do not start training:

```bash
# Shared imports, paths, permissions and filesystem operations; no GPU work or copying.
bash /increase_kairos_vepfs/increase/liwenhao/agent/2337888765/heft/latent_video/classification/baselines/vjepa2/pool_launch.sh --check

# Shared runtime and tiny CUDA checks; no environment or weight copies.
bash /increase_kairos_vepfs/increase/liwenhao/agent/2337888765/heft/latent_video/classification/baselines/vjepa2/pool_launch.sh --prepare-only

# Resume the last completed epoch, keeping the same mount, source and configuration.
bash /increase_kairos_vepfs/increase/liwenhao/agent/2337888765/heft/latent_video/classification/baselines/vjepa2/pool_launch.sh \
  --resume /increase_kairos_vepfs/increase/liwenhao/agent/2337888765/runs/ssv2_vjepa2_vitl_singlehead/latest.pt
```

`VJEPA_OUTPUT_DIR` overrides the output directory for a new run. `VJEPA_TMPDIR`
selects a short node-local path for Unix sockets and multiprocessing temporary
files. Do not supply the old developer container's `/dev/shm/.../venv` path on a new
node. New nodes use the mounted shared environment directly.

Default persistent output:
`/increase_kairos_vepfs/increase/liwenhao/agent/2337888765/runs/ssv2_vjepa2_vitl_singlehead/`.
`pool_*.log` includes preparation and the entire run; `launch_*.log` covers the
inner launch. An image/pod deployment failure can occur before either log exists;
inspect platform events in that case. Model identity uses content hashes, while
data/source mounts must retain their shared paths.

SSv2 remains on shared storage. The redundant per-video path resolution has been
removed. Development-node loading measurements do not establish the training
node's shared-filesystem throughput.

All launches read the currently mounted source, without cloning or pulling Git.
Retain that source version while a run and its checkpoint evaluations are in use.
