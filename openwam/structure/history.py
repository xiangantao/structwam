"""Per-step raw camera history for the structure student at deploy time.

The student reads the last ``length`` frames of each camera (33 for the VACE student), at the
native control rate.  ``WAMPolicy`` pushes every observation — not only the ones that trigger a
replan — so the buffer has no gaps.  Before the episode has ``length`` frames the first frame is
repeated, the same padding the training reader uses at episode starts.
"""

from collections import deque
from typing import Optional

import numpy as np

CAMERAS = ("head", "wrist")


def to_uint8(image, size: int) -> np.ndarray:
    """PIL image / HWC array -> ``(size, size, 3)`` uint8."""
    from PIL import Image

    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    if image.mode != "RGB":
        image = image.convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


class StructureHistory:
    def __init__(self, length: int, size: int = 256):
        self.length, self.size = int(length), int(size)
        self._frames = {camera: deque(maxlen=self.length) for camera in CAMERAS}

    def reset(self) -> None:
        for frames in self._frames.values():
            frames.clear()

    def push(self, raw_images: Optional[dict]) -> None:
        if not raw_images:
            return
        for camera in CAMERAS:
            image = raw_images.get(camera)
            if image is None:
                continue
            frames = self._frames[camera]
            frame = to_uint8(image, self.size)
            if not frames:
                frames.extend([frame] * (self.length - 1))
            frames.append(frame)

    def frames(self) -> Optional[dict]:
        """``{camera: (length, size, size, 3) uint8}`` for cameras seen so far, else ``None``."""
        out = {camera: np.stack(frames) for camera, frames in self._frames.items() if len(frames) == self.length}
        return out or None
