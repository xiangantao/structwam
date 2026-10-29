"""Per-sample image augmentation shared by every view of one camera (B1 / B2 runs).

One parameter draw per sample is applied identically to all of its images: the video frames
that go into the VAE canvas, the first-frame condition, and the structure student's 33-frame
history — so the world model, the action stream and the student always see the same scene.

* head camera: a small random affine (zoom, shift, in-plane rotation) and, with some probability,
  a Gaussian blur — 2-D stand-ins for small camera moves and sensor degradation;
* both cameras: one brightness / contrast / saturation (/ hue) change (a global lighting change), or
  with ``independent_color`` a separate draw per camera (pi0.5 / Cosmos-Policy style).

The affine is defined in normalized image coordinates ([-1, 1] on both axes), so it is the same
transform at any resolution or aspect ratio (the canvas head slot is 256x320, the student's frames
256x256).  The zoom is never smaller than what the rotation and shift need, so no border is ever
exposed.  Random numbers come from ``random`` (seeded per DataLoader worker by PyTorch).
"""

import math
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass
class AugmentConfig:
    enabled: bool = False
    rotate_deg: float = 4.0        # head: max |in-plane rotation|
    scale_max: float = 1.2         # head: max zoom (>= 1; the minimum is set by the rotation)
    brightness: float = 0.25       # both cameras: factor in [1 - x, 1 + x]
    contrast: float = 0.25
    saturation: float = 0.3
    blur_prob: float = 0.2         # head: probability of a Gaussian blur
    blur_radius: float = 1.5       # head: max blur radius in pixels at 256 px width
    hue: float = 0.0               # hue shift in [-x, x], fraction of the hue circle (0 = off)
    independent_color: bool = False  # separate color draws for head and wrist

    @classmethod
    def from_any(cls, value) -> "AugmentConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        items = value.items() if hasattr(value, "items") else dict(value).items()
        known = {k: v for k, v in items if k in cls.__dataclass_fields__}
        return cls(**known)


class ConsistentAugment:
    def __init__(self, config: AugmentConfig):
        self.config = config

    def _color(self) -> dict:
        c = self.config
        return {"brightness": random.uniform(1 - c.brightness, 1 + c.brightness),
                "contrast": random.uniform(1 - c.contrast, 1 + c.contrast),
                "saturation": random.uniform(1 - c.saturation, 1 + c.saturation)}

    def sample(self) -> dict:
        # Draw order is fixed so the default config reproduces earlier runs' random streams exactly;
        # hue and the per-camera draws only consume random numbers when enabled.
        c = self.config
        theta = math.radians(random.uniform(-c.rotate_deg, c.rotate_deg))
        cover = abs(math.cos(theta)) + abs(math.sin(theta))     # zoom needed to hide rotated corners
        scale = random.uniform(cover, max(cover, c.scale_max))
        shift = scale / cover - 1.0                              # remaining margin for the shift
        params = {
            "theta": theta,
            "scale": scale,
            "shift": (random.uniform(-shift, shift), random.uniform(-shift, shift)),
            **self._color(),
            "blur": random.uniform(0.3, c.blur_radius) if random.random() < c.blur_prob else 0.0,
        }
        params["hue"] = random.uniform(-c.hue, c.hue) if c.hue > 0 else 0.0
        if c.independent_color:
            wrist = self._color()
            wrist["hue"] = random.uniform(-c.hue, c.hue) if c.hue > 0 else 0.0
            params["wrist_color"] = wrist
        return params

    @staticmethod
    def _shift_hue(img: Image.Image, hue: float) -> Image.Image:
        h, s, v = img.convert("HSV").split()
        offset = int(round(hue * 255))
        h = h.point(lambda x: (x + offset) % 256)
        return Image.merge("HSV", (h, s, v)).convert("RGB")

    @staticmethod
    def _affine(image: Image.Image, p: dict) -> Image.Image:
        """Output pixel -> input pixel for p_out = scale * R(theta) * p_in + shift (normalized coords)."""
        w, h = image.size
        cos, sin = math.cos(-p["theta"]) / p["scale"], math.sin(-p["theta"]) / p["scale"]
        m00, m01, m10, m11 = cos, -sin, sin, cos                 # R(-theta) / scale
        tx, ty = p["shift"]
        a, b = m00, m01 * w / h
        c = w / 2 * (m00 * (-1 - tx) + m01 * (-1 - ty)) + w / 2
        d, e = m10 * h / w, m11
        f = h / 2 * (m10 * (-1 - tx) + m11 * (-1 - ty)) + h / 2
        return image.transform((w, h), Image.AFFINE, (a, b, c, d, e, f), resample=Image.BILINEAR)

    def apply(self, image, p: dict, camera: str):
        """``image``: PIL image or uint8 HxWx3 array (returned in the same type)."""
        is_array = isinstance(image, np.ndarray)
        img = Image.fromarray(image) if is_array else image
        mode = img.mode
        if mode != "RGB":
            img = img.convert("RGB")
        if camera == "head":
            img = self._affine(img, p)
            if p["blur"] > 0:
                img = img.filter(ImageFilter.GaussianBlur(p["blur"] * img.size[0] / 256))
        color = p["wrist_color"] if camera == "wrist" and "wrist_color" in p else p
        img = ImageEnhance.Brightness(img).enhance(color["brightness"])
        img = ImageEnhance.Contrast(img).enhance(color["contrast"])
        img = ImageEnhance.Color(img).enhance(color["saturation"])
        if color.get("hue", 0.0):
            img = self._shift_hue(img, color["hue"])
        if mode != "RGB":
            img = img.convert(mode)
        return np.asarray(img, dtype=np.uint8) if is_array else img

    def apply_clip(self, clip: np.ndarray, p: dict, camera: str) -> np.ndarray:
        """uint8 ``[T, H, W, 3]`` -> same shape."""
        return np.stack([self.apply(frame, p, camera) for frame in clip])


def build(value) -> Optional[ConsistentAugment]:
    config = AugmentConfig.from_any(value)
    return ConsistentAugment(config) if config.enabled else None
