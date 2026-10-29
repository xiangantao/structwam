#!/usr/bin/env python3
"""Scan every VACE cross-attention head for a full-instruction target readout.

The teacher runs its normal protocol (observed pass at t=1000 and the future capture
after the sparse denoise update), but every ``attn2`` of the 30 main blocks and the 15
VACE blocks is captured.  Each head's attention to the *whole* instruction is reduced to
one 32x32 map per readout, without choosing words:

* ``native_mass``: attention mass on the instruction tokens (padding keys included in the
  softmax, as the model computes it);
* ``content_snmax`` / ``content_snmean``: per content token (stop words removed), its map
  divided by its own spatial mean, then max / mean over tokens (``snmax`` is the readout
  the teacher uses today, on L22H4 only);
* ``spatial_all``: every instruction token's softmax over cells, averaged over tokens;
* ``eos``: the end-of-sentence token's softmax over cells.

Each map is kept for the present latent frame (``now``) and averaged over the last four
latent frames (``recent``).  Scoring against simulator GT is done separately
(``report_vace_full_prompt_heads.py``); nothing here reads GT.

``--build-windows`` writes the window list (several starts per held-out tracking episode);
then run one process per NPU with ``--rank r --world n``.
"""

import argparse
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

if os.environ.get("VACE_NPU", "0") == "1":
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT / "scripts"))

READOUTS = ("native_mass", "content_snmax", "content_snmean", "spatial_all", "eos")
TIMES = ("recent", "now")
CAMERA_TAGS = {"observation.images.image": "image", "observation.images.image2": "image2"}
WINDOW = 49
CURRENT = {}


def log(**values):
    print(json.dumps({"time": round(time.time(), 1), **values}), flush=True)


def swap_prompt(window, prompts, tracking_root):
    """Another instruction of the same suite for the same window (control only).

    ``clean``: it names at least one object in this scene and nothing of the category of the
    object that the episode moves next (GT, used only to pick the control).  Otherwise any other
    instruction of the suite (e.g. libero_spatial, where every instruction names a bowl).
    """
    from report_vace_full_prompt_heads import ground_truth, head_noun, mentioned_groups

    root = Path(tracking_root)
    metadata = root / "videos" / f"episode_{window['tracking_episode']:05d}" / "metadata.json"
    names = json.loads(metadata.read_text())["body_names"]
    gt, _ = ground_truth(window, root)
    target = gt["target_name"] if gt else None
    others = sorted(p for p in prompts if p != window["task"])
    named = {p: set(mentioned_groups(p, names)) for p in others}
    clean = [p for p in others if named[p] and target is not None
             and head_noun(target) not in {head_noun(g) for g in named[p]}]
    rng = np.random.default_rng(window["teacher_seed"])
    if clean:
        most = max(len(named[p]) for p in clean)
        best = [p for p in clean if len(named[p]) == most]
        return best[int(rng.integers(len(best)))], True
    return others[int(rng.integers(len(others)))], False


def build_windows(heldout, output, starts_per_episode, tracking_root=None):
    rows = [json.loads(line) for line in Path(heldout).read_text().splitlines() if line.strip()]
    episodes = {}
    for row in rows:
        episodes.setdefault((row["episode"], row["camera"]), row)
    prompts = {}
    for row in rows:
        prompts.setdefault(row["suite"], set()).add(row["task"])
    windows = []
    for (episode, camera), row in sorted(episodes.items()):
        last = (row["length"] - WINDOW) // 4 * 4          # GT is annotated every 4 frames
        if last < 0:
            raise ValueError(f"Episode {episode} is shorter than one window")
        starts = sorted({int(round(value / 4)) * 4 for value in np.linspace(0, last, starts_per_episode)})
        for start in starts:
            sample_id = f"e{episode}-{CAMERA_TAGS[camera]}-s{start:04d}"
            window = {
                "sample_id": sample_id, "episode": episode, "camera": camera, "suite": row["suite"],
                "task": row["task"], "task_index": row["task_index"], "tracking_episode": row["tracking_episode"],
                "video": row["video"], "fps": row["fps"], "length": row["length"], "start": start,
                "frame_indices": [row["episode_start_frame"] + start + i for i in range(WINDOW)],
                "teacher_seed": zlib.crc32(sample_id.encode()) & 0x7FFFFFFF,
            }
            windows.append(window)
            if tracking_root:
                # Same video and noise seed; only the instruction differs.
                task, clean = swap_prompt(window, prompts[row["suite"]], tracking_root)
                windows.append({**window, "sample_id": sample_id + "-swap", "task": task,
                                "swap_of": sample_id, "swap_clean": clean})
    output.mkdir(parents=True, exist_ok=True)
    (output / "windows.jsonl").write_text("".join(json.dumps(w) + "\n" for w in windows))
    log(stage="windows", episodes=len(episodes), windows=len(windows),
        clean_swaps=sum(bool(w.get("swap_clean")) for w in windows))


class ScanCapture:
    """Drop-in for ``teacher.Capture``: reduces every attn2 head to full-instruction maps."""

    def __init__(self, model, task, temporal, grid, text_mask):
        self.model, self.temporal, self.grid = model, temporal, grid
        self.mask = text_mask.bool()
        if self.mask.ndim == 1:
            self.mask = self.mask[None]
        # Future latent (49 frames, 13 latents): the present frame 32 is latent 8.
        # Observed latent: the last latent is the most recent one.
        self.slots = [5, 6, 7, 8] if temporal == 13 else list(range(temporal - 4, temporal))
        self.content = CURRENT["content"]
        self.values, self.modules = {}, []

    def _blocks(self):
        main = [(f"m{i:02d}", block.attn2) for i, block in enumerate(self.model.blocks)]
        vace = [] if CURRENT.get("main_only") else \
            [(f"v{i:02d}", block.attn2) for i, block in enumerate(self.model.vace_blocks)]
        return main + vace

    def __enter__(self):
        for name, attention in self._blocks():
            attention._heft_capture = self._hook(name)
            self.modules.append(attention)
        return self

    def __exit__(self, *_):
        for attention in self.modules:
            del attention._heft_capture
        missing = [name for name, _ in self._blocks() if name not in self.values]
        if missing:
            raise RuntimeError(f"attn2 capture did not fire for {missing}")

    def _hook(self, name):
        def capture(query, key, **_):
            batch, heads, tokens, dim = query.shape
            cells = tokens // self.temporal
            if cells != self.grid**2 or batch != len(self.mask):
                raise RuntimeError(f"Unexpected attn2 shape {tuple(query.shape)} in {name}")
            q = query.reshape(batch, heads, self.temporal, cells, dim)[:, :, self.slots].float()
            out = []
            for b in range(batch):
                valid = self.mask[b].to(query.device)
                content = torch.as_tensor(self.content[b], device=query.device)
                if (content & ~valid).any() or not content.any():
                    raise RuntimeError("Content tokens must be a non-empty subset of the instruction")
                eos = int(valid.nonzero().max())
                logits = torch.einsum("hscd,hld->hscl", q[b], key[b].float()) / math.sqrt(dim)
                native = logits.softmax(-1)[..., valid].sum(-1)
                words = logits[..., valid]
                share = words.softmax(-1)[..., content[valid]]
                relative = share / share.mean(2, keepdim=True).clamp_min(1e-8)
                spatial = words.softmax(2).mean(-1)
                end = logits[..., eos].softmax(2)
                maps = torch.stack((native, relative.amax(-1), relative.mean(-1), spatial, end))
                maps = maps / maps.sum(-1, keepdim=True).clamp_min(1e-12)        # [R, H, S, C]
                out.append(torch.stack((maps.mean(2), maps[:, :, -1]), 1))      # [R, 2, H, C]
            self.values[name] = torch.stack(out).half().cpu()
        return capture

    def result(self, index=None):
        names = [name for name, _ in self._blocks()]
        maps = torch.stack([self.values[name] for name in names], 1)            # [B, blocks, R, 2, H, C]
        return {"maps": maps if index is None else maps[index], "temporal": torch.tensor(self.temporal)}


@torch.no_grad()
def run(args):
    from vace_latent_distill import teacher as teacher_module
    from vace_latent_distill.config import NEGATIVE_PROMPT, PipelineConfig
    from vace_latent_distill.data import decode_indices

    rows = [json.loads(line) for line in (args.output / "windows.jsonl").read_text().splitlines()]
    config = PipelineConfig.load(args.config)
    maps_dir = args.output / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    todo = [row for row in rows[args.rank::args.world] if not (maps_dir / f"{row['sample_id']}.npz").is_file()]
    log(stage="start", rank=args.rank, windows=len(rows[args.rank::args.world]), todo=len(todo))
    if not todo:
        return
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    text_cache = torch.load(config.train.text_cache, map_location="cpu", weights_only=True)
    content = teacher_module.content_token_masks(config.teacher.text_root, sorted({r["task"] for r in rows}))
    teacher_module.Capture = ScanCapture
    teacher_module.validate_target = lambda *_, **__: None
    teacher = teacher_module.VaceTeacher(config.teacher, "semantic", args.device, text_cache[NEGATIVE_PROMPT])
    CURRENT["main_only"] = args.main_only
    names = [f"m{i:02d}" for i in range(len(teacher.model.blocks))] + \
            ([] if args.main_only else [f"v{i:02d}" for i in range(len(teacher.model.vace_blocks))])
    try:
        for begin in range(0, len(todo), args.batch_size):
            batch = todo[begin:begin + args.batch_size]
            tick = time.monotonic()
            videos = [torch.from_numpy(decode_indices(r["video"], r["frame_indices"], r["fps"])) for r in batch]
            texts = [text_cache[r["task"]] for r in batch]
            CURRENT["content"] = [content[r["task"]] for r in batch]
            targets = teacher.extract_batch(videos, texts, [t.abs().sum(-1) != 0 for t in texts],
                                            [r["teacher_seed"] for r in batch], CURRENT["content"])
            for row, target in zip(batch, targets, strict=True):
                path = maps_dir / f"{row['sample_id']}.npz"
                temporary = path.with_name(path.stem + ".tmp.npz")
                np.savez(temporary, observed=target["observed"]["maps"].numpy(),
                         future=target["future"]["maps"].numpy(), blocks=np.array(names),
                         readouts=np.array(READOUTS), times=np.array(TIMES),
                         observed_temporal=int(target["observed"]["temporal"]),
                         future_temporal=int(target["future"]["temporal"]))
                temporary.replace(path)
            log(stage="batch", rank=args.rank, done=begin + len(batch), todo=len(todo),
                seconds=round(time.monotonic() - tick, 1))
    finally:
        teacher.close()
    if args.rank == 0:
        (args.output / "protocol.json").write_text(json.dumps(teacher.protocol, indent=2, default=str) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-windows", type=Path, help="held-out manifest to build windows.jsonl from")
    parser.add_argument("--starts", type=int, default=4, help="window starts per episode and camera")
    parser.add_argument("--swap-tracking-root", type=Path,
                        help="also add a swapped-instruction control window per window (GT root picks the swap)")
    parser.add_argument("--main-only", action="store_true", help="capture the 30 main blocks only")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.build_windows:
        return build_windows(args.build_windows, args.output, args.starts, args.swap_tracking_root)
    run(args)


if __name__ == "__main__":
    main()
