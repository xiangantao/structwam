#!/usr/bin/env python3
"""Assemble per-episode policy features into frame-indexed arrays for policy training.

Outputs (``--output``), aligned with the LeRobot dataset's global frame ``index``:

* ``objects.npy``  [N, 2, 8, 42] fp16 -- per camera (image, image2) and K block:
  0 valid | 1-2 centre | 3-5 area, box w, h | 6-11 appearance (PCA of block-mean K) |
  12-13 wrist depth, depth valid | 14-16 relevance to the task's 3 words |
  17-32 past centre displacement (frames t-32..t-4, dx dy) | 33-40 future (t+4..t+16) |
  41 tracking confidence
* ``tracks.npy``   [N, 2, 32, 12, 3] fp16 -- point displacements (dx, dy) and similarity
* ``owner.npy``    [N, 2, 32] int8 -- block slot of each point (-1 = padding)
* ``prompt.npy``   [tasks, 3, 66] fp32 -- per selected word: PCA-64 text embedding, position, valid
* ``frames.npz``   episode, frame, index, task_index, state [N, 8], action [N, 7], has_features
* ``meta.json``    column layout, PCA bases, words, source checkpoints
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CAMERAS = ("observation.images.image", "observation.images.image2")
APPEARANCE = 6
PROMPT_DIMS = 64


def pca(values, dims):
    mean = values.mean(0)
    _, _, vt = np.linalg.svd(values - mean, full_matrices=False)
    return mean.astype(np.float32), vt[:dims].astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--words", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-missing", action="store_true", help="keep frames without features (testing)")
    args = parser.parse_args()
    import pyarrow.parquet as pq
    import torch

    args.output.mkdir(parents=True, exist_ok=True)
    columns = ["episode_index", "frame_index", "index", "task_index", "observation.state", "action"]
    tables = [pq.read_table(p, columns=columns).to_pydict()
              for p in sorted(glob.glob(str(args.dataset_root / "data/chunk-*/file-*.parquet")))]
    data = {c: np.concatenate([np.asarray(t[c]) for t in tables]) for c in columns}
    order = np.argsort(data["index"])
    data = {c: v[order] for c, v in data.items()}
    count = len(data["index"])
    if not np.array_equal(data["index"], np.arange(count)):
        raise ValueError("Dataset frame index is not a contiguous 0..N-1 range")
    tasks = pq.read_table(args.dataset_root / "meta/tasks.parquet").to_pandas()
    task_names = {int(row["task_index"]): name for name, row in tasks.iterrows()} if "task_index" in tasks else {}
    lookup = {(int(e), int(f)): i for i, (e, f) in enumerate(zip(data["episode_index"], data["frame_index"]))}

    files = sorted(args.raw.glob("*.npz"))
    blocks_k = []
    rng = np.random.default_rng(0)
    for path in files:                                       # appearance PCA: sample of valid blocks
        with np.load(path) as raw:
            k = raw["k"][raw["valid"]].astype(np.float32)
        if len(k):
            blocks_k.append(k[rng.choice(len(k), min(len(k), 200), replace=False)])
    k_mean, k_basis = pca(np.concatenate(blocks_k), APPEARANCE)

    objects = np.zeros((count, 2, 8, 42), np.float16)
    tracks = np.zeros((count, 2, 32, 12, 3), np.float16)
    owner = np.full((count, 2, 32), -1, np.int8)
    has = np.zeros((count, 2), bool)
    words = json.loads(args.words.read_text())
    for path in files:
        with np.load(path) as raw:
            camera = CAMERAS.index(str(raw["camera"]))
            episode = int(raw["episode"])
            frames = len(raw["valid"])
            rows = np.asarray([lookup[(episode, t)] for t in range(frames)])
            valid = raw["valid"]
            appearance = (raw["k"].astype(np.float32) - k_mean) @ k_basis.T
            block = np.concatenate((
                valid[..., None].astype(np.float32), raw["centre"], raw["size"], appearance, raw["depth"],
                raw["relevance"], raw["trajectory"].reshape(frames, 8, 24), raw["confidence"][..., None]), -1)
            block[~valid] = 0
            objects[rows, camera] = block.astype(np.float16)
            tracks[rows, camera] = raw["tracks"]
            owner[rows, camera] = raw["owner"]
            has[rows, camera] = True
    complete = has.all(1)
    if not args.allow_missing and not complete.all():
        raise ValueError(f"{int((~complete).sum())} frames lack features for both cameras")

    text_cache = torch.load(args.text_cache, map_location="cpu", weights_only=True)
    names = sorted(words)
    vectors, places = [], []
    for name in names:
        embedding = text_cache[name].float().numpy()
        length = max(int((np.abs(embedding).sum(-1) != 0).sum()), 1)
        for word in words[name]["words"]:
            vectors.append(embedding[word].mean(0))
            places.append(word[0] / length)
    p_mean, p_basis = pca(np.stack(vectors), PROMPT_DIMS)
    by_name = {name: i for i, name in enumerate(names)}
    prompt = np.zeros((len(names), 3, PROMPT_DIMS + 2), np.float32)
    cursor = 0
    for name in names:
        for slot, _ in enumerate(words[name]["words"]):
            prompt[by_name[name], slot, :PROMPT_DIMS] = (vectors[cursor] - p_mean) @ p_basis.T
            prompt[by_name[name], slot, PROMPT_DIMS] = places[cursor]
            prompt[by_name[name], slot, PROMPT_DIMS + 1] = 1.0
            cursor += 1
    task_prompt = np.asarray([by_name.get(task_names.get(int(t), ""), -1) for t in range(max(task_names) + 1)]) \
        if task_names else np.zeros(0, int)

    np.save(args.output / "objects.npy", objects)
    np.save(args.output / "tracks.npy", tracks)
    np.save(args.output / "owner.npy", owner)
    np.save(args.output / "prompt.npy", prompt)
    np.savez(args.output / "frames.npz", episode=data["episode_index"], frame=data["frame_index"],
             index=data["index"], task_index=data["task_index"], task_prompt=task_prompt,
             state=np.stack(data["observation.state"]).astype(np.float32),
             action=np.stack(data["action"]).astype(np.float32), has_features=complete)
    meta = {"columns": {"valid": [0], "centre": [1, 2], "size": [3, 4, 5], "appearance": list(range(6, 12)),
                        "depth": [12, 13], "relevance": [14, 15, 16], "past": list(range(17, 33)),
                        "future": list(range(33, 41)), "confidence": [41]},
            "cameras": CAMERAS, "frames": count, "with_features": int(complete.sum()), "files": len(files),
            "task_order": names, "words": {n: words[n]["text"] for n in names},
            "appearance_pca": {"mean": k_mean.tolist(), "basis": k_basis.tolist()},
            "prompt_pca": {"mean": p_mean.tolist(), "basis": p_basis.tolist()}}
    (args.output / "meta.json").write_text(json.dumps(meta) + "\n")
    print(json.dumps({k: meta[k] for k in ("frames", "with_features", "files")}))


if __name__ == "__main__":
    main()
