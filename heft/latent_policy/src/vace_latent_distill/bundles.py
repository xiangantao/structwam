"""Read named region bundles alongside an unfiltered full-scene field."""

import importlib

import numpy as np

from .config import FUTURE_ANCHORS, OBSERVED_ANCHORS


def phrase_positions(tokenizer, prompt, phrases):
    tokens = tokenizer(prompt, padding="max_length", max_length=512, truncation=True)["input_ids"]
    masks = {}
    for phrase in dict.fromkeys(phrases):
        ids = tokenizer(phrase, add_special_tokens=False)["input_ids"]
        positions = [index + offset
                     for index in range(len(tokens) - len(ids) + 1)
                     if tokens[index:index + len(ids)] == ids
                     for offset in range(len(ids))]
        if not ids or not positions:
            raise ValueError(f"Phrase is absent from the instruction: {phrase}")
        mask = np.zeros(len(tokens), bool)
        mask[positions] = True
        masks[phrase] = mask
    return masks


def word_heats(attention, content_mask, phrases):
    attention = np.asarray(attention, np.float32)
    content_mask = np.asarray(content_mask, bool)
    if not content_mask.any() or content_mask.shape != attention.shape[-1:]:
        raise ValueError("A matching nonempty content-token mask is required")
    maps = [attention[..., content_mask].max(-1)]
    for mask in phrases.values():
        mask = np.asarray(mask, bool)
        if mask.shape != content_mask.shape or not mask.any():
            raise ValueError("A matching nonempty phrase mask is required")
        maps.append(attention[..., mask].sum(-1))
    return np.stack(maps)


def trajectory_density(paths, valid, grid):
    """Bilinear point occupancy, with separate anchor and point identities retained."""
    if paths.shape[:-1] != valid.shape or paths.shape[-1] != 2:
        raise ValueError("Paths and temporal-support masks differ")
    bundles, anchors, points, temporal = valid.shape
    density = np.zeros((bundles, anchors, temporal, grid, grid), np.float32)
    for bundle in range(bundles):
        for anchor in range(anchors):
            for time in range(temporal):
                xy = paths[bundle, anchor, valid[bundle, anchor, :, time], time]
                if not len(xy):
                    continue
                xy = np.clip(xy, 0, grid - 1)
                left = np.floor(xy).astype(int)
                delta = xy - left
                for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                    weight = (delta[:, 0] if dx else 1 - delta[:, 0]) * (
                        delta[:, 1] if dy else 1 - delta[:, 1])
                    cells = np.minimum(left + (dx, dy), grid - 1)
                    np.add.at(density[bundle, anchor, time], (cells[:, 1], cells[:, 0]),
                              weight / len(xy))
    return density


def decode_paths(q, k, fields, seeds, anchors, config, *, public=False):
    demo = importlib.import_module("demo_main_prompt_k_tracking")
    temporal, cells = q.shape[:2]
    grid = round(cells**0.5)
    paths = np.zeros((len(anchors), len(seeds[0]), temporal, 2), np.float32)
    raw_paths = np.zeros_like(paths)
    valid = np.zeros(paths.shape[:-1], bool)
    for index, anchor in enumerate(anchors):
        end = min(anchor + 3, temporal)
        region = np.ones((temporal, cells), np.float32) if public else (
            demo.k_region_over_time(fields, seeds[index], anchor, grid))
        raw = demo.track_dense(
            q[:end].reshape(-1, q.shape[-1]), k[:end].reshape(-1, k.shape[-1]),
            seeds[index], anchor, grid, region[:end],
            topk=config.beam_topk, beam_size=config.beam_size,
            max_step=config.beam_max_step, motion_weight=config.motion_weight,
            local_weight=config.local_weight, region_weight=config.region_weight,
            acceleration_weight=config.acceleration_weight, descriptor_ema=config.descriptor_ema,
            direction="q_to_k",
        )
        paths[index, :, anchor:end] = demo.regularize_paths(
            raw, config.regularize_max_step, config.regularize_max_acceleration, passes=2)
        raw_paths[index, :, anchor:end] = raw
        valid[index, :, anchor:end] = True
    return paths, raw_paths, valid


def read_region_bundles(branch, name, phrases, content_mask, config, progress=None):
    from .teacher import local_scripts

    local_scripts()
    demo = importlib.import_module("demo_main_prompt_k_tracking")
    anchors = OBSERVED_ANCHORS if name == "observed" else FUTURE_ANCHORS
    fields = {22: branch["seg_k_l22h9"].float().numpy(),
              15: branch["seg_k_l15h7"].float().numpy()}
    q, k = branch["track_q"].float().numpy(), branch["track_k"].float().numpy()
    temporal, cells = q.shape[:2]
    grid = round(cells**0.5)
    heat = word_heats(branch["text_attention"].float().numpy(), content_mask, phrases)
    heat = heat.reshape(len(heat), temporal, grid, grid)
    regions, selected_seeds, paths, raw_paths, support = [], [], [], [], []
    for label, bundle_heat in zip(("Full instruction", *phrases), heat, strict=True):
        scores, selections = zip(*[demo.region_from_prompt_and_k(bundle_heat, fields, time, grid)
                                  for time in range(temporal)], strict=True)
        seeds = np.stack([selected[np.argsort(scores[anchor].ravel()[selected])[
            -min(config.anchors_per_time, len(selected)):]]
            for anchor, selected in ((anchor, selections[anchor]) for anchor in anchors)])
        decoded, raw, valid = decode_paths(q, k, fields, seeds, anchors, config)
        regions.append(np.stack(scores))
        selected_seeds.append(seeds)
        paths.append(decoded)
        raw_paths.append(raw)
        support.append(valid)
        if progress:
            progress(branch=name, bundle=label, points_per_anchor=len(seeds[0]))
    paths, raw_paths, support = np.stack(paths), np.stack(raw_paths), np.stack(support)
    density = trajectory_density(paths, support, grid)
    public_seeds = np.broadcast_to(np.arange(cells), (len(anchors), cells)).copy()
    public_paths, public_raw, public_valid = decode_paths(
        q, k, fields, public_seeds, anchors, config, public=True)
    return {
        "prompt_heat": heat, "region_heat": np.stack(regions),
        "seeds": np.stack(selected_seeds), "paths": paths, "raw_paths": raw_paths,
        "path_valid": support, "trajectory_heat_by_anchor": density,
        "trajectory_heat": density.max(1), "anchor_native": np.asarray(anchors),
        "public_seeds": public_seeds, "public_paths": public_paths,
        "public_raw_paths": public_raw, "public_path_valid": public_valid,
    }
