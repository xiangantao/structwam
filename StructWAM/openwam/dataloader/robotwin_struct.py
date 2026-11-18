"""RoboTwin 2.0 reader for StructWAM: the canonical RoboTwin sample plus what the structure stream needs.

Same windows, canvas, actions and normalization as ``openwam.dataloader.robotwin`` (the OpenWAM
RoboTwin protocol); added per sample:

* ``structure_frames``: ``{"head": uint8 [H, S, S, 3], "wrist": ..., "wrist_right": ...}`` — the ``H``
  (33) raw frames of each camera ending at the window's first frame (the present), at the native
  rate, stretched to the student's square resolution (as the canvas stretches each camera into its
  tile).  Frames before the episode start repeat the first frame.  Structure camera -> HDF5 camera
  is ``structure_cameras`` (default head_camera / left_camera / right_camera, i.e. the canvas
  slots [head, bottom-left, bottom-right]).
* ``phase`` / ``phase_chunk``: contact phase ``rho`` at the present and its maximum over the action
  window.  Bimanual: ``rho = max(rho_left, rho_right)``, each from that arm's gripper command
  (RoboTwin: 1 open .. 0 closed, thresholded at 0.5).

``structure_augment`` (train split, off by default): one ``openwam.structure.augment`` draw per
sample, applied identically to every camera's canvas frames and to the student's history (head:
affine + optional blur, all cameras: one color change), at the native resolution before any resize.
"""

from typing import Any, Optional

import h5py
import numpy as np
from PIL import Image

from openwam.dataloader.robotwin import MultiTaskRoboTwinDataset, RoboTwinDataset
from openwam.dataloader.transforms.multiview import assemble_multiview_layout
from openwam.structure import augment as augment_lib
from openwam.structure.phase import contact_phase

DEFAULT_STRUCTURE_CAMERAS = {"head": "head_camera", "wrist": "left_camera", "wrist_right": "right_camera"}
GRIPPER_OPEN_THRESHOLD = 0.5


def _augment_kind(structure_camera: str) -> str:
    """augment.apply's camera kind: head (affine + blur + color) or wrist (color only)."""
    return "head" if structure_camera == "head" else "wrist"


class RoboTwinStructDataset(RoboTwinDataset):
    def __init__(self, data_root: str, *, structure_history: int = 33, structure_size: int = 256,
                 phase_sigma: float = 8.0, structure_augment: Any = None,
                 structure_cameras: Optional[dict] = None, **kwargs: Any):
        self._structure_history = int(structure_history)
        self._structure_size = int(structure_size)
        self._phase_sigma = float(phase_sigma)
        self._structure_cameras = dict(structure_cameras or DEFAULT_STRUCTURE_CAMERAS)
        if "head" not in self._structure_cameras:
            raise ValueError(f"structure_cameras needs a head camera, got {self._structure_cameras}")
        self._kind_of_hdf5 = {cam: _augment_kind(name) for name, cam in self._structure_cameras.items()}
        self._phase_cache: dict = {}
        self._augment = augment_lib.build(structure_augment)
        self._aug_params = None
        super().__init__(data_root, **kwargs)

    # ----- augmentation (canvas frames) --------------------------------------------

    def _read_camera_frames(self, f, camera_key: str, start: int, end: int):
        frames = super()._read_camera_frames(f, camera_key, start, end)
        if self._aug_params is not None and frames:
            kind = self._kind_of_hdf5.get(camera_key, "wrist")
            frames = [self._augment.apply(frame, self._aug_params, kind) for frame in frames]
        return frames

    def _read_multiview_frames(self, f, cameras, start, end):
        """As the base reader, but only the canvases ``_build_sample`` keeps are decoded and composed:
        the ``video_stride`` frames and the last one (tail padding).  The others are ``None``.
        (The base reader composes all 33 canvases of 3 cameras to keep 9.)"""
        n = end - start
        keep = sorted({i for i in self._video_sample_indices if i < n} | {n - 1})
        per_camera = {}
        for cam in cameras:
            path = f"observation/{cam}/rgb"
            if cam is not None and path in f:
                raw = f[path][start:end]
                images = {i: self._decode_jpeg(raw[i]) for i in keep}
                if self._aug_params is not None:
                    kind = self._kind_of_hdf5.get(cam, "wrist")
                    images = {i: self._augment.apply(image, self._aug_params, kind) for i, image in images.items()}
            else:
                ph, pw = self._obs_image_size or (self.height, self.width)
                images = {i: Image.new("RGB", (pw, ph), (0, 0, 0)) for i in keep}
            per_camera[cam] = images
        canvases = [None] * n
        for i in keep:
            canvases[i] = assemble_multiview_layout({cam: per_camera[cam][i] for cam in cameras}, self.camera_layout,
                                                    self.height, self.width)
        return canvases

    # ----- structure frames -------------------------------------------------------

    def _history(self, f, name: str, camera: str, present: int) -> Optional[np.ndarray]:
        path = f"observation/{camera}/rgb"
        if path not in f:
            return None
        local = np.clip(np.arange(present - self._structure_history + 1, present + 1), 0, None)
        lo = int(local.min())
        raw = f[path][lo : present + 1]
        decoded = {}
        size = self._structure_size
        kind = _augment_kind(name)
        for i in sorted(set(local.tolist())):
            image = self._decode_jpeg(raw[i - lo])
            if self._aug_params is not None:
                image = self._augment.apply(image, self._aug_params, kind)
            decoded[i] = np.asarray(image.resize((size, size), Image.BILINEAR), dtype=np.uint8)
        return np.ascontiguousarray(np.stack([decoded[i] for i in local.tolist()]))

    # ----- contact phase ------------------------------------------------------------

    def _episode_phase(self, ep_idx: int, f) -> np.ndarray:
        if ep_idx not in self._phase_cache:
            arms = [contact_phase(f[f"endpose/{arm}_gripper"][:].astype(np.float32) - GRIPPER_OPEN_THRESHOLD,
                                  self._phase_sigma) for arm in ("left", "right")]
            self._phase_cache[ep_idx] = np.maximum(*arms)
        return self._phase_cache[ep_idx]

    # ----- sample ---------------------------------------------------------------------

    def __getitem__(self, idx):
        self._aug_params = self._augment.sample() if self._augment is not None and self.split == "train" else None
        try:
            sample = super().__getitem__(idx)
            ep_idx, start = self._window_index[idx]
            with h5py.File(self._episode_files[ep_idx], "r") as f:
                frames = {}
                for name, camera in self._structure_cameras.items():
                    clip = self._history(f, name, camera, start)
                    if clip is not None:
                        frames[name] = clip
                rho = self._episode_phase(ep_idx, f)
        finally:
            self._aug_params = None
        if "head" not in frames:
            raise RuntimeError(f"RoboTwin-struct: no head-camera history in {self._episode_files[ep_idx]}")
        end = min(len(rho), start + self.num_frames)
        sample["structure_frames"] = frames
        sample["phase"] = float(rho[min(start, len(rho) - 1)])
        sample["phase_chunk"] = float(rho[start:end].max()) if end > start else sample["phase"]
        return sample


class MultiTaskRoboTwinStructDataset(MultiTaskRoboTwinDataset):
    _SUB_DATASET_CLS = RoboTwinStructDataset
    _EXTRA_CONFIG_KEYS = ("structure_history", "structure_size", "phase_sigma", "structure_augment",
                          "structure_cameras")

    @classmethod
    def from_config(cls, config, split: str = "train"):
        # Hydra DictConfigs -> plain containers for the per-task readers (and their worker pickling).
        try:
            from omegaconf import DictConfig, OmegaConf

            if isinstance(config, DictConfig):
                config = OmegaConf.to_container(config, resolve=True)
        except ImportError:
            pass
        return super().from_config(config, split=split)
