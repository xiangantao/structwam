"""MoT driver for the StructWAM tri-system: ``TriSystemMoTDriver`` plus four changes.

1. Structure reads video: the structure tail may attend to video keys (the tri-system default
   keeps the tail read-only on itself), so the world model's imagination can correct the
   student's longer-range tracks.  Structure never reads action tokens.
2. Phase routing: when a gate is set for the forward, the action query rows get a float mask
   with ``RoutingBias`` added (see ``openwam.structure.routing``); all other rows keep the
   boolean mask.
3. Alignment capture: the video hidden state after ``align_layer`` is kept for the
   representation-alignment loss (taken from the step's *output*, so it is valid under
   gradient checkpointing).
4. Video losses train the structure expert by default: ``video_detach`` is empty,
   so video queries read structure keys / values with gradients intact. Optional
   ``video_detach`` entries ("structure", "action") detach those streams only for video
   query rows. With only "structure" detached, video losses can still reach the structure
   expert indirectly through action tokens; adding "action" also blocks that path.
5. Video-free rows (opt-in, ``set_forward_context(video_drop=...)``): for the marked samples
   the action rows *and* the structure rows get no video keys, so the action comes from the
   structure tokens, proprio and text alone (structure rows would otherwise relay the video).

The joint attention is computed per row group — video, action, structure — each with its own
mask rows and K/V: the forward values equal one masked attention over the whole sequence, but
the large video / structure groups use boolean masks (fused flash attention on Ascend; a float
mask over the whole sequence is ~6x slower there) and only the few action rows carry the float
routing bias, whose gradient reaches the gate and the routing weights.
"""

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from openwam.model.architectures.tri_system.mot_driver import TriSystemMoTDriver


class StructMoTDriver(TriSystemMoTDriver):
    def __init__(self, vb, ab, sb, *, routing=None, structure_reads_video: bool = True,
                 align_layer: Optional[int] = None, video_detach: Sequence[str] = (), **kwargs):
        super().__init__(vb, ab, sb, **kwargs)
        unknown = set(video_detach) - {"structure", "action"}
        if unknown:
            raise ValueError(f"video_detach accepts 'structure' / 'action', got {sorted(unknown)}")
        self.routing = routing
        self.structure_reads_video = bool(structure_reads_video)
        self.align_layer = align_layer
        self.video_detach = frozenset(video_detach)
        self._sizes: Optional[tuple] = None
        self._action_mask: Optional[Tensor] = None
        self._gate: Optional[Tensor] = None
        self._wrist_video: Optional[Tensor] = None
        self._video_drop: Optional[Tensor] = None
        self.captured_hidden: Optional[Tensor] = None

    def set_forward_context(self, gate: Optional[Tensor], wrist_video: Optional[Tensor] = None,
                            video_drop: Optional[Tensor] = None) -> None:
        """Per-forward routing input, set before the joint loop; ``gate=None`` keeps the plain boolean mask.

        The action-row mask built from it stays on the driver until the next forward builds its
        mask: activation checkpointing recomputes the attention during backward and must see
        the same mask.  ``video_drop`` (bool ``[B]``) marks the samples whose action and structure
        rows see no video keys; ``None`` or all-False leaves the masks unchanged.
        """
        self._gate, self._wrist_video = gate, wrist_video
        self._video_drop = video_drop if video_drop is not None and bool(video_drop.any()) else None
        self.captured_hidden = None

    def _build_attention_mask(self, s_video, s_action, s_understanding, video_tokens_per_frame, *, device,
                              und_mask=None):
        self._sizes = (s_video, s_action, s_understanding)
        mask = super()._build_attention_mask(s_video, s_action, s_understanding, video_tokens_per_frame,
                                             device=device, und_mask=und_mask)
        if self.structure_reads_video:
            mask = mask.clone()
            mask[..., s_video + s_action :, :s_video] = True
        self._action_mask = None
        if self._gate is not None and self.routing is not None:
            self._action_mask = self.routing(mask[..., s_video : s_video + s_action, :], self._gate, s_video=s_video,
                                             s_action=s_action, wrist_video=self._wrist_video, dtype=self.vb_dtype())
        if self._video_drop is not None:
            self._action_mask = self._action_rows_without_video(mask, s_video, s_action)
        return mask

    def _action_rows_without_video(self, mask: Tensor, s_video: int, s_action: int) -> Tensor:
        """Float action-row mask ``[B, 1, s_action, S]`` (routing bias kept) with every video key
        removed for the ``video_drop`` samples; action keys stay visible, so no row is empty."""
        dtype = self.vb_dtype()
        floor = torch.finfo(dtype).min / 2
        drop = self._video_drop
        action = self._action_mask
        if action is None:
            rows = mask[..., s_video : s_video + s_action, :]
            if rows.ndim == 2:
                rows = rows.unsqueeze(0).unsqueeze(0)
            rows = rows.expand(drop.shape[0], 1, *rows.shape[-2:])
            action = torch.zeros(rows.shape, dtype=dtype, device=rows.device).masked_fill(~rows, floor)
        action = action.clone()
        action[drop.to(action.device), :, :, :s_video] = floor
        return action

    def _structure_rows_without_video(self, rows: Tensor, prefix: int, s_video: int, batch: int) -> Tensor:
        """Boolean structure-row mask with the video keys removed for the ``video_drop`` samples
        (``prefix`` K/V columns, if any, stay visible)."""
        if rows.ndim == 2:
            rows = rows.unsqueeze(0).unsqueeze(0)
        rows = rows.expand(batch, 1, *rows.shape[-2:]).clone()
        rows[self._video_drop.to(rows.device), :, :, prefix : prefix + s_video] = False
        return rows

    def vb_dtype(self) -> torch.dtype:
        return next(self.vb.parameters()).dtype

    def _mixed_attention(self, q_cat: Tensor, k_cat: Tensor, v_cat: Tensor, attn_mask: Optional[Tensor]) -> Tensor:
        detach = self.video_detach and torch.is_grad_enabled()
        if self._sizes is None or (self._action_mask is None and not detach):
            return super()._mixed_attention(q_cat, k_cat, v_cat, attn_mask)
        s_video, s_action, s_structure = self._sizes
        # Key columns: [prefix K/V (Cosmos3 only) + video | action | structure]; query rows: [video | action | structure].
        k_video = k_cat.shape[1] - s_action - s_structure
        prefix = k_video - s_video

        def rows(begin: int, end: int) -> Optional[Tensor]:
            return None if attn_mask is None else attn_mask[..., begin:end, :]

        def video_keys(x: Tensor) -> Tensor:
            if not detach:
                return x
            parts, start = [], 0
            for name, (begin, end) in (("action", (k_video, k_video + s_action)),
                                       ("structure", (k_video + s_action, x.shape[1]))):
                if name in self.video_detach:
                    parts += [x[:, start:begin], x[:, begin:end].detach()]
                    start = end
            parts.append(x[:, start:])
            return torch.cat(parts, 1)

        action_mask = rows(s_video, s_video + s_action)
        if self._action_mask is not None:
            action_mask = self._action_mask
            if prefix > 0:  # prefix K/V columns are visible to every query row
                action_mask = F.pad(action_mask, (prefix, 0))
        outputs = [
            super()._mixed_attention(q_cat[:, :s_video], video_keys(k_cat), video_keys(v_cat), rows(0, s_video)),
            super()._mixed_attention(q_cat[:, s_video : s_video + s_action], k_cat, v_cat, action_mask),
        ]
        if s_structure:
            structure_mask = rows(s_video + s_action, q_cat.shape[1])
            if self._video_drop is not None:
                if structure_mask is None:
                    raise ValueError("video_drop needs an attention mask")
                structure_mask = self._structure_rows_without_video(structure_mask, prefix, s_video, q_cat.shape[0])
            outputs.append(super()._mixed_attention(q_cat[:, s_video + s_action :], k_cat, v_cat, structure_mask))
        return torch.cat(outputs, 1)

    def step(self, layer_id, vstate, astate, ustate, attn_mask=None, *, use_gradient_checkpointing=False,
             use_gradient_checkpointing_offload=False):
        vstate, astate, ustate = super().step(
            layer_id, vstate, astate, ustate, attn_mask=attn_mask,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        if self.align_layer is not None and layer_id == self.align_layer:
            self.captured_hidden = vstate.hidden_states
        return vstate, astate, ustate


__all__ = ["StructMoTDriver"]
