"""Frame samples for the latent Diffusion Policy, read from ``finalize_policy_features.py`` output.

A sample at frame ``t``: proprio states at ``t-1, t`` (DP's two observation steps, clamped at
the episode start); the encoder features at ``t`` only (the student already summarizes its
own 33-frame history); the task's words; and the action chunk ``t-1 .. t+14`` (DP horizon 16
from the first observation step), copy-padded past the episode end.  As in DP, the last
``drop_n_last_frames`` frames of each episode are not sampled.  State and action are
min-max normalized to [-1, 1]; block and point features are z-scored per camera and column
(statistics from the training frames only); padding blocks / points stay zero.
"""

import json
from pathlib import Path

import numpy as np
import torch

# Dataset task_index ranges of the four LIBERO suites in the local LeRobot dataset.
SUITES = {"libero_10": range(0, 10), "libero_goal": range(10, 20), "libero_object": range(20, 30),
          "libero_spatial": range(30, 40)}
EPS = 1e-8


class Features:
    """Memory-mapped feature arrays shared by train/validation datasets."""

    def __init__(self, root):
        root = Path(root)
        self.objects = np.load(root / "objects.npy", mmap_mode="r")
        self.tracks = np.load(root / "tracks.npy", mmap_mode="r")
        self.owner = np.load(root / "owner.npy", mmap_mode="r")
        self.prompt = np.load(root / "prompt.npy")
        frames = np.load(root / "frames.npz")
        self.frames = {k: frames[k] for k in frames.files}
        self.meta = json.loads((root / "meta.json").read_text())


def suite_frames(features, suite, episodes=None, exclude=(), drop_n_last_frames=7):
    """Global frame indices to sample for ``suite`` (optionally restricted / excluding episodes)."""
    frames = features.frames
    keep = np.isin(frames["task_index"], list(SUITES[suite])) & frames["has_features"]
    if episodes is not None:
        keep &= np.isin(frames["episode"], list(episodes))
    keep &= ~np.isin(frames["episode"], list(exclude))
    last = np.zeros_like(keep)
    for episode in np.unique(frames["episode"][keep]):
        mine = np.flatnonzero(frames["episode"] == episode)
        last[mine[max(len(mine) - drop_n_last_frames, 1):]] = True
    return np.flatnonzero(keep & ~last)


def compute_stats(features, indices, sample=60000, seed=0):
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(indices, min(sample, len(indices)), replace=False))
    frames = features.frames
    episodes = np.unique(frames["episode"][indices])
    rows = np.flatnonzero(np.isin(frames["episode"], episodes))
    stats = {"state_min": frames["state"][rows].min(0), "state_max": frames["state"][rows].max(0),
             "action_min": frames["action"][rows].min(0), "action_max": frames["action"][rows].max(0)}
    objects = np.asarray(features.objects[pick], np.float32)                   # [S, 2, 8, 42]
    valid = objects[..., 0] > 0.5
    mean, std = np.zeros((2, 42), np.float32), np.ones((2, 42), np.float32)
    for camera in range(2):
        rows_c = objects[:, camera][valid[:, camera]]
        mean[camera], std[camera] = rows_c.mean(0), rows_c.std(0) + 1e-6
    mean[:, 0], std[:, 0] = 0.0, 1.0                                           # keep the valid flag
    tracks = np.asarray(features.tracks[pick], np.float32)                     # [S, 2, 32, 12, 3]
    alive = np.asarray(features.owner[pick]) >= 0
    t_mean, t_std = np.zeros((2, 3), np.float32), np.ones((2, 3), np.float32)
    for camera in range(2):
        points = tracks[:, camera][alive[:, camera]].reshape(-1, 3)
        t_mean[camera], t_std[camera] = points.mean(0), points.std(0) + 1e-6
    stats.update(object_mean=mean, object_std=std, track_mean=t_mean, track_std=t_std)
    return {k: np.asarray(v, np.float32) for k, v in stats.items()}


def minmax(value, low, high):
    return np.where(high - low > EPS, 2 * (value - low) / np.maximum(high - low, EPS) - 1, 0.0)


class LatentDPDataset(torch.utils.data.Dataset):
    def __init__(self, features, indices, suite, stats, n_obs_steps=2, horizon=16):
        self.f, self.indices, self.stats = features, np.asarray(indices), stats
        self.n_obs_steps, self.horizon = n_obs_steps, horizon
        self.offset = SUITES[suite].start
        frames = features.frames
        episode = frames["episode"]
        change = np.flatnonzero(np.diff(episode)) + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [len(episode)]))
        self.start = np.repeat(starts, ends - starts)
        self.end = np.repeat(ends, ends - starts)                              # exclusive

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        start, end, s = self.start[index], self.end[index], self.stats
        frames = self.f.frames
        obs = np.clip(np.arange(index - self.n_obs_steps + 1, index + 1), start, end - 1)
        span = np.arange(index - self.n_obs_steps + 1, index - self.n_obs_steps + 1 + self.horizon)
        actions = np.clip(span, start, end - 1)
        objects = np.asarray(self.f.objects[index], np.float32)
        valid = objects[..., 0] > 0.5
        objects = np.where(valid[..., None], (objects - s["object_mean"][:, None]) / s["object_std"][:, None], 0)
        owner = np.asarray(self.f.owner[index], np.int64)
        tracks = np.asarray(self.f.tracks[index], np.float32)
        tracks = np.where((owner >= 0)[..., None, None],
                          (tracks - s["track_mean"][:, None, None]) / s["track_std"][:, None, None], 0)
        task = int(frames["task_index"][index])
        prompt = self.f.prompt[int(frames["task_prompt"][task])]
        return {
            "objects": torch.from_numpy(objects.astype(np.float32)),
            "object_valid": torch.from_numpy(valid),
            "tracks": torch.from_numpy(tracks.astype(np.float32)),
            "owner": torch.from_numpy(owner),
            "prompt": torch.from_numpy(prompt.astype(np.float32)),
            "word_valid": torch.from_numpy(prompt[:, -1] > 0.5),
            "state": torch.from_numpy(minmax(frames["state"][obs], s["state_min"], s["state_max"]).astype(np.float32)),
            "task": torch.tensor(task - self.offset, dtype=torch.long),
            "action": torch.from_numpy(minmax(frames["action"][actions], s["action_min"], s["action_max"])
                                       .astype(np.float32)),
            "action_is_pad": torch.from_numpy((span < start) | (span >= end)),
            "index": torch.tensor(index),
        }
