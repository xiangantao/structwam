"""Policy maps: image-aligned single-meaning channels (A) and filtered keypoint tracks (C).

Everything comes from one camera's student fields for one window plus the instruction
text; no future frames, simulator data or external models.

* Instruction roles (``instruction_roles``): fixed grammar rules split the instruction
  into at most two sub-goals.  The object of the verb is the target; the phrase after
  on / in / into / on top of / to the X of ... is the destination; between / next to /
  from / on / in after a picked object is a locator; "it" and a bare "inside" refer back
  to the previous clause.
* Phrase heat (``phrase_heat``): the text-attention head gives, per present cell, a
  distribution over instruction tokens; a phrase's heat is the mass on its tokens over
  its spatial mean (how many times more than average this cell attends to the phrase).
* Dense motion (``dense_tracks``): K-K soft-argmax flow between consecutive native steps
  of the track K (observed frames 0..28, present 32, predicted 36..48), chained from
  every present cell, with forward-backward consistency and match distinctiveness.
* Filter (``window_maps``): reliable cells; camera motion removed on the wrist view;
  moving cells; K segments; roles hand / target / destination.  The hand is found without
  tracking: on the wrist view, the K that stays put while the view changes; on the third
  person view, where the K at a fixed position changes over +-8 frames.  Target and
  destination are the K segments with the highest phrase heat.  Everything else is zero.
* Word mode (``word_window_maps``, no grammar): instead of the roles, the instruction's
  content words ranked by attention concentration; up to five above a threshold give word
  maps, and their best K segments take the place of the target / destination segments.
  The role mode (``window_maps``) is kept for the existing DP runs.
"""

import numpy as np
import torch
from torch.nn import functional as F

GRID = 32
STEPS = 13                     # observed natives 4..11 (frames 0..28), future slots 8..12 (frames 32..48)
PRESENT = 8                    # sequence index of the present frame (future slot 8)
HORIZONS = (0, 4, 6, 10, 12)   # frames -32, -16, -8, +8, +16 from the present
RADIUS, TEMPERATURE = 3, 0.02
FB_LIMIT = 0.5                 # cells (4 px at 256)
MOVE_MIN = 0.75                # cells (6 px)
STILL_MIN, STILL_MARGIN = 0.75, 0.12   # wrist view: zero-offset K similarity over time (gripper)
CHANGE_SEGMENT, CHANGE_CELL = 0.35, 0.45   # third person: K change at a fixed position over +-8 frames
BACKGROUND_AREA = {False: 0.30, True: 0.60}
ROLE_HEAT = ("target_1", "destination_1", "target_2", "destination_2", "locator")
WORD_HEAT = ("word_1", "word_2", "word_3")           # the automatically selected words (words.json)
HEAT_CHANNELS = ROLE_HEAT + WORD_HEAT
APPEARANCE = 6
MAP_CHANNELS = (tuple(f"{h}_now" for h in HEAT_CHANNELS) + tuple(f"{h}_+16" for h in HEAT_CHANNELS)
                + tuple(f"{a}{t:+d}" for t in (-32, -16, -8, 8, 16) for a in ("dx", "dy"))
                + ("motion", "reliability") + ("camera_dx-8", "camera_dy-8", "camera_dx+8", "camera_dy+8")
                + ("hand", "target", "destination", "kept") + tuple(f"k_pca{i}" for i in range(APPEARANCE))
                + ("depth",))
ROLES = ("hand", "target", "destination")
# Word mode (no grammar): the most concentrated content words, at most WORD_SLOTS, above a
# camera-specific concentration (peak log heat).  The thresholds cut off the least concentrated
# fifth of the words, whose peaks almost never land on an object (held-out windows, sim GT
# used for this choice only: third person 4% vs ~35% above, wrist 13% vs ~45%).
WORD_SLOTS = 5
WORD_MIN = {False: 0.64, True: 0.48}
WORD_HEAT_SLOTS = tuple(f"word_{i + 1}" for i in range(WORD_SLOTS))
WORD_ROLES = ("hand", "words_now", "words_+16")
WORD_KEYPOINTS = {"hand": 12, "words_now": 10, "words_+16": 10}
WORD_MAP_CHANNELS = (tuple(f"{w}_now" for w in WORD_HEAT_SLOTS) + tuple(f"{w}_+16" for w in WORD_HEAT_SLOTS)
                     + tuple(f"{a}{t:+d}" for t in (-32, -16, -8, 8, 16) for a in ("dx", "dy"))
                     + ("motion", "reliability") + ("camera_dx-8", "camera_dy-8", "camera_dx+8", "camera_dy+8")
                     + ("hand", "words_now", "words_+16", "kept") + tuple(f"k_pca{i}" for i in range(APPEARANCE))
                     + ("depth",))
KEYPOINTS = {"hand": 12, "target": 10, "destination": 10}
KEYPOINT_SIMILARITY = 0.6

VERBS = {"pick", "place", "put", "open", "close", "turn", "push"}
DROP = {"the", "a", "an", "both", "and", "of"}
STOP = {"the", "a", "an", "and", "to", "of", "on", "in", "at", "is", "it", "up"}   # as the teacher's content mask
DESTINATION = (("on", "top", "of"), ("to", "the", "front", "of"), ("to", "the", "back", "of"),
               ("to", "the", "right", "of"), ("to", "the", "left", "of"), ("in", "front", "of"),
               ("into",), ("onto",), ("inside",), ("on",), ("in",), ("to",))
LOCATOR = (("next", "to"), ("between",), ("from",), ("on",), ("in",), ("at",))


# ----------------------------------------------------------------------------- instruction

def instruction_words(task, tokenizer, length=512):
    """Words of the instruction (lower case) and their token indices in the padded prompt."""
    pieces = tokenizer.convert_ids_to_tokens(
        tokenizer(task, padding="max_length", max_length=length, truncation=True)["input_ids"])
    special = set(tokenizer.all_special_tokens)
    words, tokens = [], []
    for index, piece in enumerate(pieces):
        if piece in special:
            continue
        text = piece.replace(chr(9601), "").lower()
        if piece.startswith(chr(9601)) or not words:
            if not text:              # a bare word-start marker; the next piece begins the word
                words.append("")
                tokens.append([index])
                continue
            words.append(text)
            tokens.append([index])
        else:
            words[-1] += text
            tokens[-1].append(index)
    keep = [i for i, w in enumerate(words) if w.strip(".,")]
    return [words[i].strip(".,") for i in keep], [tokens[i] for i in keep]


def _find(words, start, end, patterns):
    for i in range(start, end):
        for pattern in patterns:
            if tuple(words[i:min(i + len(pattern), end)]) == pattern:
                return i, len(pattern)
    return None, 0


def _phrase(words, start, end, head=True):
    span = list(range(start, end))
    if head and "of" in words[start:end]:
        span = span[: words[start:end].index("of")]
    return [i for i in span if words[i] not in DROP]


def _clause(words, start, end):
    verb, i = words[start], start + 1
    if i < end and words[i] in ("up", "on", "off"):
        i += 1
    clause = {"verb": verb, "target": [], "destination": [], "locator": [], "target_it": False,
              "destination_it": False}
    if verb == "pick":
        j, n = _find(words, i, end, LOCATOR)
        clause["target"] = _phrase(words, i, end if j is None else j)
        if j is not None:
            clause["locator"] = _phrase(words, j + n, end, head=False)
    elif verb in ("put", "place", "push"):
        j, n = _find(words, i, end, DESTINATION)
        clause["target"] = _phrase(words, i, end if j is None else j)
        if j is not None:
            clause["destination"] = _phrase(words, j + n, end)
            clause["destination_it"] = not clause["destination"] or [words[k] for k in clause["destination"]] == ["it"]
    else:
        clause["target"] = _phrase(words, i, end)
    clause["target_it"] = [words[k] for k in clause["target"]] == ["it"]
    return clause


def instruction_roles(words):
    """Sub-goals [{target, destination}] (word indices, at most two) and locator word indices."""
    cuts = [i for i in range(len(words) - 1) if words[i] == "and" and words[i + 1] in VERBS]
    bounds = list(zip([0] + [c + 1 for c in cuts], cuts + [len(words)]))
    goals, locator = [], []
    for start, end in bounds:
        clause = _clause(words, start, end)
        locator += clause["locator"]
        previous = goals[-1] if goals else None
        target, destination = clause["target"], clause["destination"]
        if clause["target_it"] and previous is not None:
            if clause["verb"] in ("close", "open"):
                target = previous["destination"] or previous["target"]
            elif not previous["destination"]:          # "pick up X ... and place it in Y": one sub-goal
                previous["destination"] = destination
                continue
            else:
                target = previous["target"]
        if clause["destination_it"] and previous is not None:
            destination = previous["target"]
        goals.append({"target": target, "destination": destination})
    return goals[:2], locator


def role_tokens(task, tokenizer, selected=None):
    """{channel name: token indices} for ``HEAT_CHANNELS`` plus the readable words.

    ``selected``: the task's entry of words.json (automatically selected words), if any.
    """
    words, tokens = instruction_words(task, tokenizer)
    goals, locator = instruction_roles(words)
    result, readable = {name: [] for name in HEAT_CHANNELS}, {}
    for g, goal in enumerate(goals):
        for role in ("target", "destination"):
            name = f"{role}_{g + 1}"
            result[name] = [t for w in goal[role] for t in tokens[w]]
            readable[name] = " ".join(words[w] for w in goal[role])
    result["locator"] = [t for w in locator for t in tokens[w]]
    readable["locator"] = " ".join(words[w] for w in locator)
    for i, (word, text) in enumerate(zip((selected or {}).get("words", []), (selected or {}).get("text", []))):
        result[WORD_HEAT[i]] = list(word)
        readable[WORD_HEAT[i]] = text
    result["content"] = [t for w, ts in zip(words, tokens) if w not in STOP for t in ts]
    return result, readable


def phrase_heat(attention, tokens, content=None):
    """``attention`` [cells, L] -> log spatially-normalized heat [cells] (0 where no tokens).

    With ``content`` (token indices of the instruction's content words) each cell's attention
    is first renormalized over those words: which word is this cell about, rather than how
    much it attends to the instruction at all.
    """
    attention = np.asarray(attention, np.float32)
    if not tokens:
        return np.zeros(len(attention), np.float32)
    if content:
        attention = attention / np.maximum(attention[:, content].sum(1, keepdims=True), 1e-8)
    heat = attention[:, tokens].sum(1)
    return np.log(np.maximum(heat / max(float(heat.mean()), 1e-8), 1e-3)).astype(np.float32)


# ----------------------------------------------------------------------------- dense motion

def _offsets(device):
    size = 2 * RADIUS + 1
    values = torch.arange(size, device=device).float() - RADIUS
    dy, dx = torch.meshgrid(values, values, indexing="ij")
    return torch.stack((dx.flatten(), dy.flatten()), -1)                     # [window, 2] (dx, dy)


def match(a, b):
    """Unit fields [W, G, G, C] at two times -> for each cell of ``a``: flow to ``b`` [W, G, G, 2]
    (cells, x/y), peak similarity and distinctiveness (peak minus the best outside the peak's 3x3)."""
    size = 2 * RADIUS + 1
    count, channels = a.shape[0], a.shape[-1]
    windows = F.unfold(b.permute(0, 3, 1, 2), size, padding=RADIUS).reshape(count, channels, size * size, GRID * GRID)
    similarity = torch.einsum("wcn,wckn->wkn", a.reshape(count, GRID * GRID, channels).transpose(1, 2), windows)
    inside = F.unfold(torch.ones(1, 1, GRID, GRID, device=a.device), size, padding=RADIUS) > 0     # [1, window, cells]
    similarity = similarity.masked_fill(~inside, -2.0)
    offsets = _offsets(a.device)
    weights = (similarity / TEMPERATURE).softmax(1)
    flow = torch.einsum("wkn,kd->wnd", weights, offsets)                                        # [W, cells, 2]
    peak, best = similarity.max(1)
    near = (offsets[:, None, None] - offsets[best][None]).abs().max(-1).values <= 1             # [window, W, cells]
    second = similarity.masked_fill(near.permute(1, 0, 2), -2.0).max(1).values
    shape = (count, GRID, GRID)
    return flow.reshape(*shape, 2), peak.reshape(shape), (peak - second).reshape(shape)


def sample(field, positions):
    """Bilinear [W, G, G, ...] at cell positions [W, N, 2] (x, y) -> [W, N, ...]."""
    count = field.shape[0]
    flat = field.reshape(count, GRID, GRID, -1).permute(0, 3, 1, 2).float()
    grid = ((positions + 0.5) / GRID * 2 - 1)[:, :, None]
    value = F.grid_sample(flat, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return value[..., 0].transpose(1, 2).reshape(count, positions.shape[1], *field.shape[3:])


def dense_tracks(track_k):
    """``track_k`` [13, cells, C] or [W, 13, cells, C] (sequence order) -> chained tracks from
    every present cell (batched over windows; same leading shape as the input).

    Returns dict of tensors: ``positions`` [(W,) 13, cells, 2] in cells (x, y), ``valid``
    (forward-backward consistent on every step from the present), ``fb``, and ``peak`` /
    ``distinct`` of the step that reached that time, each [(W,) 13, cells].
    """
    track_k = torch.as_tensor(track_k)
    single = track_k.ndim == 3
    fields = F.normalize(track_k.float().reshape(-1, STEPS, GRID, GRID, track_k.shape[-1]), dim=-1)
    count, device = fields.shape[0], fields.device
    ys, xs = torch.meshgrid(torch.arange(GRID, device=device), torch.arange(GRID, device=device), indexing="ij")
    start = torch.stack((xs.flatten(), ys.flatten()), -1).float()[None].expand(count, -1, -1)
    cells = GRID * GRID
    positions = torch.zeros(count, STEPS, cells, 2, device=device)
    valid = torch.zeros(count, STEPS, cells, dtype=torch.bool, device=device)
    fb = torch.zeros(count, STEPS, cells, device=device)
    peak = torch.ones(count, STEPS, cells, device=device)
    distinct = torch.ones(count, STEPS, cells, device=device)
    positions[:, PRESENT], valid[:, PRESENT] = start, True
    for direction in (-1, 1):
        position, ok = start.clone(), torch.ones(count, cells, dtype=torch.bool, device=device)
        k = PRESENT
        while 0 <= k + direction < STEPS:
            flow_ab, peak_ab, distinct_ab = match(fields[:, k], fields[:, k + direction])
            flow_ba = match(fields[:, k + direction], fields[:, k])[0]
            step = sample(flow_ab, position)
            reached = position + step
            error = (step + sample(flow_ba, reached)).norm(dim=-1)
            inside = ((reached >= -0.5) & (reached <= GRID - 0.5)).all(-1)
            ok = ok & (error < FB_LIMIT) & inside
            k += direction
            positions[:, k], valid[:, k], fb[:, k] = reached, ok, error
            peak[:, k], distinct[:, k] = sample(peak_ab, position), sample(distinct_ab, position)
            position = reached
    result = {"positions": positions, "valid": valid, "fb": fb, "peak": peak, "distinct": distinct}
    return {k: v[0] for k, v in result.items()} if single else result


def fixed_position(track_k):
    """K similarity at a fixed image position: ``still`` [(W,) cells] = mean over the other
    times of cos(K_t, K_present); ``change`` = 1 - mean cos(K_present, K at +-8 frames)."""
    k = F.normalize(torch.as_tensor(track_k).float(), dim=-1)
    similarity = (k * k[..., PRESENT:PRESENT + 1, :, :]).sum(-1)                    # [(W,) 13, cells]
    others = [i for i in range(STEPS) if i != PRESENT]
    still = similarity[..., others, :].mean(-2)
    change = 1 - 0.5 * (similarity[..., PRESENT - 2, :] + similarity[..., PRESENT + 2, :])
    return still, change


def camera_motion(displacement, reliable, iterations=4, keep=0.7):
    """Robust affine fit of displacement [N, 2] over present positions -> fitted field [N, 2]."""
    ys, xs = np.mgrid[0:GRID, 0:GRID]
    design = np.stack((xs.ravel(), ys.ravel(), np.ones(GRID * GRID)), -1) / np.array([GRID, GRID, 1])
    chosen = np.flatnonzero(reliable)
    if len(chosen) < 12:
        return np.zeros_like(displacement)
    for _ in range(iterations):
        coefficients, *_ = np.linalg.lstsq(design[chosen], displacement[chosen], rcond=None)
        residual = np.linalg.norm(design[chosen] @ coefficients - displacement[chosen], axis=-1)
        chosen = chosen[residual <= np.quantile(residual, keep)]
    return design @ coefficients


# ----------------------------------------------------------------------------- segments

def segments(seg_k_l22h9, seg_k_l15h7, wrist):
    """K-means + connected components of the present K -> list of dicts (cells, area, border)."""
    from scipy import ndimage

    from vace_latent_distill.policy_features import kmeans

    labels = kmeans(present_k(seg_k_l22h9, seg_k_l15h7)).numpy().reshape(GRID, GRID)
    result = []
    for k in np.unique(labels):
        components, count = ndimage.label(labels == k)
        for c in range(1, count + 1):
            mask = components == c
            ys, xs = np.nonzero(mask)
            area = mask.mean()
            result.append({"mask": mask.ravel(), "area": float(area),
                           "border": bool(xs.min() == 0 or ys.min() == 0 or xs.max() == GRID - 1 or ys.max() == GRID - 1),
                           "background": bool(area > BACKGROUND_AREA[bool(wrist)])})
    return result


def _fps(cells, score, count):
    """Farthest point sampling over cell indices, starting at the highest ``score``."""
    if len(cells) <= count:
        return list(cells)
    xy = np.stack((cells % GRID, cells // GRID), -1).astype(np.float32)
    chosen = [int(np.argmax(score))]
    distance = np.linalg.norm(xy - xy[chosen[0]], axis=-1)
    while len(chosen) < count:
        chosen.append(int(np.argmax(distance)))
        distance = np.minimum(distance, np.linalg.norm(xy - xy[chosen[-1]], axis=-1))
    return [int(cells[i]) for i in chosen]


# ----------------------------------------------------------------------------- window

def present_k(seg_k_l22h9, seg_k_l15h7):
    """Present K of both segmentation heads, unit halves -> [cells, 256]."""
    return torch.cat((F.normalize(torch.as_tensor(seg_k_l22h9).float(), dim=-1),
                      F.normalize(torch.as_tensor(seg_k_l15h7).float(), dim=-1)), -1).cpu() / np.sqrt(2)


def _window_core(fields, wrist):
    """Motion, segments and the hand of one camera window (shared by the role and word modes)."""
    if "dense" in fields:                                                      # precomputed (batched on device)
        dense, still_map, change_map = fields["dense"], fields["still"], fields["change"]
    else:
        dense = {k: v.cpu().numpy() for k, v in dense_tracks(fields["track_k"]).items()}
        still_map, change_map = (v.cpu().numpy() for v in fixed_position(fields["track_k"]))
    positions, valid = dense["positions"], dense["valid"]
    displacement = positions - positions[PRESENT][None]                       # [13, cells, 2] cells
    reliable = valid[PRESENT - 1] & valid[PRESENT + 1] & (dense["distinct"][PRESENT + 1] > 0.02)
    camera = np.zeros_like(displacement)
    if wrist:
        for k in range(STEPS):
            if k != PRESENT:
                camera[k] = camera_motion(displacement[k], reliable & valid[k])
    residual = np.where(valid[..., None], displacement - camera, 0.0)
    step_size = np.linalg.norm(residual, axis=-1)                              # [13, cells] cells
    magnitude = step_size.max(0)
    # Chained tracks drift with distance from the present: one noise level per step.
    noise = np.array([np.median(step_size[k][reliable & valid[k]]) if (reliable & valid[k]).any() else 0.0
                      for k in range(STEPS)])
    threshold = np.maximum(MOVE_MIN, 3.0 * noise)
    moving = reliable & ((step_size > threshold[:, None]) & valid).any(0)
    parts = segments(fields["seg_k_l22h9"], fields["seg_k_l15h7"], wrist)

    moving_parts = np.zeros(GRID * GRID, bool)
    for part in parts:
        cells = part["mask"] & reliable
        if part["background"] or cells.sum() < 2:
            continue
        if moving[part["mask"]].sum() >= 2 and moving[cells].mean() >= 0.4:
            moving_parts |= part["mask"]
    if wrist:
        # The gripper rides with the wrist camera: it keeps the same K at the same image
        # position while the rest of the view changes (works on textureless fingers too).
        still = still_map
        level = float(np.median(still))
        hand = np.zeros(GRID * GRID, bool)
        for part in parts:
            score = float(still[part["mask"]].mean())
            if part["mask"].sum() >= 3 and score > max(STILL_MIN, level + STILL_MARGIN):
                hand |= part["mask"]
    else:
        # Third person: the gripper (and what it holds) is where the K at a fixed position
        # changes between the present and +-8 frames; tracking fails there (fast, deforming).
        change = change_map
        hand = change > CHANGE_CELL
        for part in parts:
            if not part["background"] and change[part["mask"]].mean() > CHANGE_SEGMENT:
                hand |= part["mask"]
        still = change
    return {"dense": dense, "positions": positions, "valid": valid, "displacement": displacement,
            "camera": camera, "residual": residual, "magnitude": magnitude, "threshold": threshold,
            "reliable": reliable, "moving": moving, "parts": parts, "moving_parts": moving_parts,
            "hand": hand, "still": still}


def _best_segment(parts, heat):
    """The non-background K segment (>= 2 cells) with the highest mean heat, or ``None``."""
    scores = [(heat[p["mask"]].mean(), i) for i, p in enumerate(parts) if not p["background"] and p["mask"].sum() >= 2]
    return parts[max(scores)[1]]["mask"] if scores else None


def _fill_motion_maps(maps, core, kept, wrist, fields, k_basis):
    """Flow at five horizons, motion, reliability, camera motion, depth and appearance PCA."""
    displacement, valid = core["displacement"], core["valid"]
    scale = 4.0                                                                # 4 cells (32 px) -> 1.0
    for index, offset in zip(HORIZONS, (-32, -16, -8, 8, 16)):
        ok = kept & valid[index]
        maps[f"dx{offset:+d}"][ok], maps[f"dy{offset:+d}"][ok] = (displacement[index, ok] / scale).T
    maps["motion"][kept] = np.minimum(core["magnitude"][kept] / scale, 2.0)
    maps["reliability"] = valid.mean(0)
    if wrist:
        for index, offset in ((6, -8), (10, 8)):
            maps[f"camera_dx{offset:+d}"], maps[f"camera_dy{offset:+d}"] = (core["camera"][index] / scale).T
        depth = np.asarray(fields["depth"], np.float32)
        maps["depth"] = (depth - depth.mean()) / max(float(depth.std()), 1e-6)
    maps["kept"] = kept.astype(np.float32)
    if k_basis is not None:
        appearance = (present_k(fields["seg_k_l22h9"], fields["seg_k_l15h7"]).numpy() - k_basis[0]) @ k_basis[1].T
        for i in range(APPEARANCE):
            maps[f"k_pca{i}"] = appearance[:, i]


def _keypoints(core, roles, role_masks, kept, scores, budget, keypoint_tracker, wrist):
    """C: keypoints per role, farthest point sampling inside the kept role cells."""
    chosen, picked_roles = [], []
    for r, role in enumerate(roles):
        cells = np.flatnonzero(role_masks[role] & kept)
        score = core["magnitude"][cells] if role == "hand" else (scores[role][cells] if len(cells) else cells)
        picked = _fps(cells, score, budget[role]) if len(cells) else []
        chosen += picked
        picked_roles += [r] * len(picked)
    count = sum(budget.values())
    keypoints = {"xy": np.zeros((count, STEPS, 2), np.float32), "similarity": np.zeros((count, STEPS), np.float32),
                 "valid": np.zeros((count, STEPS), bool), "role": np.full(count, -1, np.int64),
                 "cell": np.full(count, -1, np.int64)}
    if chosen:
        n = len(chosen)
        cells = np.asarray(chosen)
        keypoints["xy"][:n] = core["positions"][:, cells].transpose(1, 0, 2)
        keypoints["similarity"][:n] = core["dense"]["peak"][:, cells].T
        keypoints["valid"][:n] = core["valid"][:, cells].T
        keypoints["role"][:n] = picked_roles
        keypoints["cell"][:n] = cells
        if keypoint_tracker is not None:
            refine_keypoints(keypoints, *keypoint_tracker(cells), wrist)
    return keypoints


def _diagnostics(core, role_masks, kept, heat):
    return {"dense": core["dense"], "displacement": core["displacement"], "residual": core["residual"],
            "camera": core["camera"], "magnitude": core["magnitude"], "threshold": core["threshold"],
            "still": core["still"],
            "raw_moving": (np.linalg.norm(core["displacement"], axis=-1) > core["threshold"][:, None]).any(0),
            "reliable": core["reliable"], "moving": core["moving"], "moving_parts": core["moving_parts"],
            "roles": role_masks, "kept": kept, "heat": heat, "parts": core["parts"]}


def window_maps(fields, tokens, wrist, keypoint_tracker=None, k_basis=None):
    """One camera window, role mode (grammar roles from ``role_tokens``; legacy, kept for the DP runs).

    ``fields``: student fields for this window (numpy or tensors): ``track_k`` [13, cells, C] in
    sequence order (or precomputed ``dense`` / ``still`` / ``change`` from ``dense_tracks`` and
    ``fixed_position``); ``seg_k_l22h9`` / ``seg_k_l15h7`` [cells, C] and ``depth`` [cells] at the
    present; ``text_attention`` [2, cells, L] at the present and at +16 frames (future slot 12).
    ``tokens``: ``role_tokens`` output.  ``keypoint_tracker(cells) -> (pixels [P, 13, 2],
    similarity [P, 13])`` refines C tracks (HEFT K-K); without it the chained dense tracks are
    used.  ``k_basis``: (mean [256], basis [APPEARANCE, 256]) fixed PCA of present K.
    Returns ``maps`` [len(MAP_CHANNELS), G, G], ``keypoints`` dict and diagnostics.
    """
    attention = np.asarray(fields["text_attention"], np.float32)
    heat = np.stack([[phrase_heat(attention[t], tokens[name], tokens.get("content") if t else None)
                      for name in HEAT_CHANNELS] for t in range(2)])
    core = _window_core(fields, wrist)
    role_masks = {"hand": core["hand"]}
    for role in ("target", "destination"):
        mask = np.zeros(GRID * GRID, bool)
        for g in (1, 2):
            name = f"{role}_{g}"
            if not tokens[name]:
                continue
            # Destinations are grounded only as the object approaches them: use the +16 heat.
            best = _best_segment(core["parts"], heat[1 if role == "destination" else 0, HEAT_CHANNELS.index(name)])
            if best is not None:
                mask |= best
        role_masks[role] = mask
    kept = core["hand"] | ((role_masks["target"] | role_masks["destination"]) & core["valid"][PRESENT + 1])

    maps = {name: np.zeros(GRID * GRID, np.float32) for name in MAP_CHANNELS}
    for t, suffix in enumerate(("now", "+16")):
        for h, name in enumerate(HEAT_CHANNELS):
            if tokens[name]:
                maps[f"{name}_{suffix}"] = np.clip(heat[t, h], -2.0, 3.0) / 3.0
    _fill_motion_maps(maps, core, kept, wrist, fields, k_basis)
    for role in ROLES:
        maps[role] = (role_masks[role] & kept).astype(np.float32)
    scores = {"target": heat[0, HEAT_CHANNELS.index("target_1")],
              "destination": heat[1, HEAT_CHANNELS.index("destination_1")]}
    keypoints = _keypoints(core, ROLES, role_masks, kept, scores, KEYPOINTS, keypoint_tracker, wrist)
    stacked = np.stack([maps[name] for name in MAP_CHANNELS]).reshape(len(MAP_CHANNELS), GRID, GRID)
    return stacked.astype(np.float32), keypoints, _diagnostics(core, role_masks, kept, heat)


# ----------------------------------------------------------------------------- word mode

def content_words(task, tokenizer):
    """Instruction content words (generic stop words removed; repeats merged) with their token
    indices, and all content token indices.  No grammar, no per-task lists."""
    words, tokens = instruction_words(task, tokenizer)
    merged = {}
    for word, token_ids in zip(words, tokens):
        if word and word not in STOP:
            merged.setdefault(word, []).extend(token_ids)
    entries = list(merged.items())
    return entries, [t for _, ids in entries for t in ids]


def select_words(attention, words, content, wrist):
    """Rank content words by attention concentration (peak log spatially-normalized heat on the
    present map), keep those above ``WORD_MIN`` (camera-specific), at most ``WORD_SLOTS``.

    Returns ``[(word index, concentration, heat_now, heat_+16)]`` in descending concentration.
    """
    scored = []
    for index, (_, token_ids) in enumerate(words):
        now = phrase_heat(attention[0], token_ids, content)
        later = phrase_heat(attention[1], token_ids, content)
        scored.append((index, float(now.max()), now, later))
    scored.sort(key=lambda entry: -entry[1])
    return [entry for entry in scored if entry[1] >= WORD_MIN[bool(wrist)]][:WORD_SLOTS]


def word_window_maps(fields, words, wrist, keypoint_tracker=None, k_basis=None):
    """One camera window, word mode: no grammar.

    ``words``: ``content_words`` entries.  The most concentrated words (``select_words``) fill
    ``word_k_now`` / ``word_k_+16`` in descending concentration; each selected word's best
    non-background K segment (by its present / +16 heat) forms ``words_now`` / ``words_+16``,
    which take the place of the role target / destination segments (kept cells, keypoints).
    Returns maps [len(WORD_MAP_CHANNELS), G, G], keypoints (roles ``WORD_ROLES``), diagnostics
    with ``selected`` = [(word, concentration)].
    """
    attention = np.asarray(fields["text_attention"], np.float32)
    content = [t for _, ids in words for t in ids]
    selected = select_words(attention, words, content, wrist)
    core = _window_core(fields, wrist)
    role_masks = {"hand": core["hand"]}
    for role, column in (("words_now", 2), ("words_+16", 3)):
        mask = np.zeros(GRID * GRID, bool)
        for entry in selected:
            best = _best_segment(core["parts"], entry[column])
            if best is not None:
                mask |= best
        role_masks[role] = mask
    kept = core["hand"] | ((role_masks["words_now"] | role_masks["words_+16"]) & core["valid"][PRESENT + 1])

    maps = {name: np.zeros(GRID * GRID, np.float32) for name in WORD_MAP_CHANNELS}
    for slot, (_, _, now, later) in enumerate(selected):
        maps[f"{WORD_HEAT_SLOTS[slot]}_now"] = np.clip(now, -2.0, 3.0) / 3.0
        maps[f"{WORD_HEAT_SLOTS[slot]}_+16"] = np.clip(later, -2.0, 3.0) / 3.0
    _fill_motion_maps(maps, core, kept, wrist, fields, k_basis)
    for role in WORD_ROLES:
        maps[role] = (role_masks[role] & kept).astype(np.float32)
    zero = np.zeros(GRID * GRID, np.float32)
    scores = {"words_now": np.max([e[2] for e in selected], 0) if selected else zero,
              "words_+16": np.max([e[3] for e in selected], 0) if selected else zero}
    keypoints = _keypoints(core, WORD_ROLES, role_masks, kept, scores, WORD_KEYPOINTS, keypoint_tracker, wrist)
    stacked = np.stack([maps[name] for name in WORD_MAP_CHANNELS]).reshape(len(WORD_MAP_CHANNELS), GRID, GRID)
    heat = np.stack([[e[2] for e in selected], [e[3] for e in selected]]) if selected else np.zeros((2, 0, GRID * GRID))
    diagnostics = _diagnostics(core, role_masks, kept, heat)
    diagnostics["selected"] = [(words[i][0], score) for i, score, _, _ in selected]
    diagnostics["selected_tokens"] = [words[i][1] for i, _, _, _ in selected]
    return stacked.astype(np.float32), keypoints, diagnostics


def refine_keypoints(keypoints, pixels, similarity, wrist):
    """Replace the chained tracks of the used keypoints by HEFT K-K tracks (``track_many``
    pixels [P, 13, 2] / similarity [P, 13] for the first P used keypoints), in place."""
    n = len(pixels)
    xy = (np.asarray(pixels, np.float32) - 3.5) / 8.0                          # pixels -> cells (track_many's centres)
    similarity = np.asarray(similarity, np.float32)
    # A step is valid while the descriptor still matches, outward from the present.
    ok = similarity >= KEYPOINT_SIMILARITY
    ok[:, PRESENT] = True
    ok[:, PRESENT::-1] = np.cumprod(ok[:, PRESENT::-1], 1).astype(bool)
    ok[:, PRESENT:] = np.cumprod(ok[:, PRESENT:], 1).astype(bool)
    if wrist:                                                                  # the gripper rides with the wrist camera
        attached = keypoints["role"][:n] == ROLES.index("hand")
        xy[attached] = xy[attached][:, PRESENT:PRESENT + 1]
        ok[attached] = True
    keypoints["xy"][:n], keypoints["similarity"][:n], keypoints["valid"][:n] = xy, similarity, ok


def keypoint_tokens(keypoints):
    """C tokens [P, 13*4 + 2 + len(ROLES)]: per step (dx, dy) from the present (cells / 4),
    similarity and validity; present position in [-1, 1]; role one-hot.  Empty slots are 0."""
    xy, present = keypoints["xy"], keypoints["xy"][:, PRESENT]
    steps = np.concatenate(((xy - present[:, None]) / 4.0, keypoints["similarity"][..., None],
                            keypoints["valid"][..., None].astype(np.float32)), -1)
    role = np.zeros((len(xy), len(ROLES)), np.float32)
    used = keypoints["role"] >= 0
    role[used, keypoints["role"][used]] = 1.0
    tokens = np.concatenate((steps.reshape(len(xy), -1), (present + 0.5) / GRID * 2 - 1, role), -1)
    tokens[~used] = 0.0
    return tokens.astype(np.float32), used
