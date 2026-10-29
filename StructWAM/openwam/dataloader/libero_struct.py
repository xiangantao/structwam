"""LIBERO reader for StructWAM: the canonical LIBERO sample plus what the structure stream needs.

Added per sample:

* ``structure_frames``: ``{"head": uint8 [H, S, S, 3], "wrist": ...}`` — the ``H`` (33) raw frames
  of each camera ending at the window's first frame (the present), at the native rate and at the
  student's resolution.  Frames before the episode start repeat the first frame.
* ``phase`` / ``phase_chunk``: contact phase ``rho`` at the present and its maximum over the
  action window, from the demonstration's own gripper command (``openwam.structure.phase``).

``structure_flip`` re-orients the student frames if this export's camera convention differs from
the one the student was trained on (LeRobot LIBERO: images rotated 180 degrees from raw sim).
"""

from typing import Any, ClassVar, Tuple

import numpy as np

from openwam.dataloader.libero import LiberoDataset
from openwam.dataloader.utils.video_io import decode_video_frames
from openwam.structure.phase import contact_phase

GRIPPER_DIM = 9
FLIPS = ("none", "hflip", "vflip", "rot180")


class LiberoStructDataset(LiberoDataset):
    DATASET_NAME = "LIBERO-struct"
    CONFIG_KEYS: ClassVar[Tuple[str, ...]] = LiberoDataset.CONFIG_KEYS + (
        "structure_history",
        "structure_size",
        "structure_flip",
        "phase_sigma",
    )

    def __init__(self, dataset_dir: str, *, structure_history: int = 33, structure_size: int = 256,
                 structure_flip: str = "none", phase_sigma: float = 8.0, **kwargs: Any):
        if structure_flip not in FLIPS:
            raise ValueError(f"structure_flip must be one of {FLIPS}, got {structure_flip!r}")
        self._structure_history = int(structure_history)
        self._structure_size = int(structure_size)
        self._structure_flip = structure_flip
        self._phase_sigma = float(phase_sigma)
        self._phase_cache: dict = {}
        super().__init__(dataset_dir, **kwargs)

    # ----- structure frames -------------------------------------------------

    def _flip(self, clip: np.ndarray) -> np.ndarray:
        if self._structure_flip == "hflip":
            return clip[:, :, ::-1]
        if self._structure_flip == "vflip":
            return clip[:, ::-1]
        if self._structure_flip == "rot180":
            return clip[:, ::-1, ::-1]
        return clip

    def _decode_history(self, camera, row, ep_local: int, present: int):
        if camera is None or camera not in self._ep_video_frame_offsets:
            return None
        chunk_col, file_col = f"videos/{camera}/chunk_index", f"videos/{camera}/file_index"
        if chunk_col not in row.index:
            return None
        path = self._dataset_dir / self._video_path_template.format(
            video_key=camera, chunk_index=int(row[chunk_col]), file_index=int(row[file_col])
        )
        local = np.clip(np.arange(present - self._structure_history + 1, present + 1), 0, None)
        base = int(self._ep_video_frame_offsets[camera][ep_local])
        unique = sorted(set((local + base).tolist()))
        size = self._structure_size
        decoded = dict(zip(unique, decode_video_frames(str(path), unique, size, size)))
        clip = np.stack([np.asarray(decoded[i].convert("RGB"), dtype=np.uint8) for i in (local + base).tolist()])
        return np.ascontiguousarray(self._flip(clip))

    # ----- contact phase -----------------------------------------------------

    def _episode_phase(self, ep_local: int, row) -> np.ndarray:
        if ep_local not in self._phase_cache:
            length = int(self._ep_valid_end[ep_local])
            table = self._load_data_table(int(row["data/chunk_index"]), int(row["data/file_index"]))
            win = table.slice(int(self._ep_data_row_offset[ep_local]), length).to_pandas()
            gripper = self._raw_action_eef10(win)[:, GRIPPER_DIM]
            self._phase_cache[ep_local] = contact_phase(gripper, self._phase_sigma)
        return self._phase_cache[ep_local]

    # ----- sample ------------------------------------------------------------

    def _getitem_impl(self, idx: int) -> dict:
        sample = super()._getitem_impl(idx)
        ep_local = int(np.searchsorted(self._cum_n_starts, idx, side="right") - 1)
        offset = int(self._ep_valid_start[ep_local]) + (idx - int(self._cum_n_starts[ep_local])) * self._window_stride
        row = self._eps_df.iloc[ep_local]
        frames = {}
        for name, camera in (("head", self._head_camera), ("wrist", self._left_wrist_camera)):
            clip = self._decode_history(camera, row, ep_local, offset)
            if clip is not None:
                frames[name] = clip
        if "head" not in frames:
            raise RuntimeError(f"{self.DATASET_NAME}: no head-camera history at idx={idx}")
        rho = self._episode_phase(ep_local, row)
        end = min(len(rho), offset + self._num_frames)
        sample["structure_frames"] = frames
        sample["phase"] = float(rho[min(offset, len(rho) - 1)])
        sample["phase_chunk"] = float(rho[offset:end].max()) if end > offset else sample["phase"]
        return sample
