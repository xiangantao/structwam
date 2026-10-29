"""StructWAM: the distilled VACE structure student as a third expert of a world-action model.

Layout (one concern per module):

* ``student``  — frozen VACE-distilled V-JEPA2 student -> per-camera structure features.
* ``history``  — per-step raw camera frame buffer used at deploy time (33 past frames).
* ``layout``   — keypoint-token layout and the multiview-canvas geometry (camera tile <-> canvas
  <-> latent / DiT token grids).
* ``tokens``   — structure features -> expert tokens (maps, keypoints, trunk).
* ``phase``    — contact-phase labels from demonstrations and the phase gate.
* ``routing``  — phase-gated attention bias (action reads structure far from contact, video and
  wrist near contact).
* ``sketch``   — structure-only coarse action chunk; warm start of the action denoiser and
  schedule shaping (few video steps, delayed action start).
* ``losses``   — structure-weighted video loss, track consistency, representation alignment.

Nothing here generates pixels: the video stream is supervised in latent space only and
``decode_video`` is always off for this architecture.
"""

from openwam.structure.history import StructureHistory
from openwam.structure.layout import CanvasLayout, KeypointLayout

__all__ = ["CanvasLayout", "KeypointLayout", "StructureHistory"]
