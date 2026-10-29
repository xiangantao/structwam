"""Coordinate-free tracking supervision and fixed-scale spatial readouts."""

import torch
from torch.nn import functional as F

from .config import HEAD_TASKS, head_view


def head_output(output, task):
    return {
        **{branch: head_view(output[branch], task) for branch in ("observed", "future")},
        **({"timeline": output["timeline"]} if "timeline" in output else {}),
    }


def masked_mean(value, valid):
    while valid.ndim < value.ndim:
        valid = valid.unsqueeze(-1)
    valid = valid.expand_as(value).bool()
    return value.masked_fill(~valid, 0).sum() / valid.sum().clamp_min(1)


def match_logits(query, key, temperature):
    return (
        F.normalize(query.float(), dim=-1)
        @ F.normalize(key.float(), dim=-1).transpose(-1, -2)
        / temperature
    )


def seed_cells(seeds, paths, path_valid, anchors, cells, min_shift):
    """Teacher seed cells and the subset whose own beam path moves.

    Returns boolean ``[B, cells]`` masks.  A seed moves when its teacher path
    leaves its anchor position by at least ``min_shift`` grid cells.  Only the
    teacher's prompt/K seeds and beam paths are used.
    """
    batch = len(seeds)
    start = torch.stack(
        [paths[:, index, :, int(anchor)] for index, anchor in enumerate(anchors.tolist())], 1
    )
    distance = (paths - start[..., None, :]).norm(dim=-1).masked_fill(~path_valid, 0)
    moving = (distance.amax(-1) >= min_shift).reshape(batch, -1).float()
    flat = seeds.reshape(batch, -1).long()
    seeded = torch.zeros(batch, cells, dtype=torch.bool, device=seeds.device)
    seeded.scatter_(1, flat, True)
    moved = torch.zeros(batch, cells, device=seeds.device).scatter_reduce(1, flat, moving, "amax")
    return seeded, moved > 0


def stratified_queries(seeded, samples, fraction):
    """Per-sample query ids ``[B, samples]``: a share drawn from seed cells first."""
    batch, cells = seeded.shape
    samples = min(samples, cells)
    from_seeds = min(round(samples * fraction), samples)
    priority = torch.rand(batch, cells, device=seeded.device)
    first = (priority + 2 * seeded).topk(from_seeds, -1).indices
    rest = torch.rand(batch, cells, device=seeded.device).scatter(1, first, -1.0)
    second = rest.topk(samples - from_seeds, -1).indices
    return torch.cat((first, second), -1)


def gather_cells(field, ids):
    return field.gather(1, ids[..., None].expand(-1, -1, field.shape[-1]))


def relation_loss(
    query, key, teacher_q, teacher_k, valid, *, samples, temperature, sampling=None, seeds=None
):
    """Teacher-relation KL between time t and t+1/t+2.

    ``sampling=None`` keeps uniform query cells shared across the batch.  With a
    ``sampling`` dict, ``seeds`` must be ``(seeded, moved)`` masks from
    ``seed_cells``: ``seed_fraction`` of each sample's queries come from teacher
    seed cells and queries on seeds whose teacher path moves get ``moving_weight``.
    """
    terms = []
    # Future native8 is the observed boundary.  Dot products never mix the two
    # different teacher extraction protocols as if their coordinates matched.
    for step in (1, 2):
        for time in range(query.shape[1] - step):
            mask = valid[:, time] & valid[:, time + step]
            if not mask.any():
                continue
            if sampling is None:
                ids = torch.randperm(query.shape[2], device=query.device)[:samples]
                pred = match_logits(query[:, time, ids], key[:, time + step], temperature)
                reference = match_logits(
                    teacher_q[:, time, ids], teacher_k[:, time + step], temperature
                )
                error = F.kl_div(
                    pred.log_softmax(-1), reference.softmax(-1), reduction="none"
                ).sum(-1)
                terms.append(masked_mean(error, mask))
                continue
            seeded, moved = seeds
            ids = stratified_queries(seeded, samples, sampling["seed_fraction"])
            pred = match_logits(gather_cells(query[:, time], ids), key[:, time + step], temperature)
            reference = match_logits(
                gather_cells(teacher_q[:, time], ids), teacher_k[:, time + step], temperature
            )
            error = F.kl_div(pred.log_softmax(-1), reference.softmax(-1), reduction="none").sum(-1)
            weight = 1 + (sampling["moving_weight"] - 1) * moved.gather(1, ids).float()
            weight = weight * mask[:, None].float()
            terms.append((error * weight).sum() / weight.sum().clamp_min(1e-8))
    if not terms:
        raise ValueError("No valid temporal pairs for tracking")
    return torch.stack(terms).mean()


@torch.no_grad()
def tracking_diagnostics(prediction, target, temperature=0.07):
    """Cheap per-branch checks that the cross-entropy alone does not show.

    ``self_match``: fraction of cells whose best t -> t+1 match is the same cell
    (teacher about 0.1-0.3 on LIBERO, a collapsed student much higher).
    ``seed_step_error``: grid-cell distance from the student's best t -> t+1 match
    to the teacher beam path, on teacher seeds at their anchors; ``seed_step_shift``
    is the student's step length there and ``teacher_step_shift`` the teacher's.
    """
    result = {}
    for branch in ("observed", "future"):
        pred, truth = prediction[branch], target[branch]
        valid = prediction["timeline"][branch]["valid"]
        anchors = prediction["timeline"][branch]["anchor_native"].tolist()
        q, k = pred["track_q"], pred["track_k"]
        tq, tk = truth["track_q"], truth["track_k"]
        cells = q.shape[2]
        grid = round(cells**0.5)
        identity = torch.arange(cells, device=q.device)
        student, teacher = [], []
        for time in range(q.shape[1] - 1):
            if not (valid[time] and valid[time + 1]):
                continue
            student.append(
                (match_logits(q[:, time], k[:, time + 1], temperature).argmax(-1) == identity)
                .float().mean()
            )
            teacher.append(
                (match_logits(tq[:, time], tk[:, time + 1], temperature).argmax(-1) == identity)
                .float().mean()
            )
        result[f"diag/{branch}/self_match_student"] = torch.stack(student).mean()
        result[f"diag/{branch}/self_match_teacher"] = torch.stack(teacher).mean()
        if "seeds" not in truth:
            continue
        errors, steps, teacher_steps = [], [], []
        for index, anchor in enumerate(anchors):
            if anchor + 1 >= q.shape[1]:
                continue
            ids = truth["seeds"][:, index].long()
            best = match_logits(gather_cells(q[:, anchor], ids), k[:, anchor + 1], temperature)
            best = best.argmax(-1)
            xy = torch.stack((best % grid, best // grid), -1).float()
            origin = torch.stack((ids % grid, ids // grid), -1).float()
            path = truth["paths"][:, index, :, anchor + 1]
            ok = truth["path_valid"][:, index, :, anchor + 1]
            errors.append((xy - path).norm(dim=-1)[ok])
            steps.append((xy - origin).norm(dim=-1)[ok])
            teacher_steps.append((path - truth["paths"][:, index, :, anchor]).norm(dim=-1)[ok])
        # Keys are always present so that ranks all-reduce the same metric list.
        for name, values in (("seed_step_error", errors), ("seed_step_shift", steps),
                             ("teacher_step_shift", teacher_steps)):
            values = torch.cat(values) if values else q.new_zeros(0)
            result[f"diag/{branch}/{name}"] = values.mean() if values.numel() else q.new_zeros(())
    return result


def cycle_loss(query, key, teacher_q, teacher_k, valid, *, samples, temperature):
    terms = []
    for time in range(query.shape[1] - 1):
        mask = valid[:, time] & valid[:, time + 1]
        if not mask.any():
            continue
        ids = torch.randperm(query.shape[2], device=query.device)[:samples]
        with torch.no_grad():
            forward = match_logits(teacher_q[:, time, ids], teacher_k[:, time + 1], temperature)
            target = forward.argmax(-1)
            batch = torch.arange(len(query), device=query.device)[:, None]
            reverse = match_logits(
                teacher_q[batch, time + 1, target], teacher_k[:, time], temperature
            )
            reliable = reverse.argmax(-1) == ids[None]
        pred_forward = match_logits(query[:, time, ids], key[:, time + 1], temperature).softmax(-1)
        pred_reverse = match_logits(query[:, time + 1], key[:, time, ids], temperature).softmax(-1)
        returned = pred_forward @ pred_reverse
        error = -returned.diagonal(dim1=-2, dim2=-1).clamp_min(1e-7).log()
        terms.append(masked_mean(error, reliable & mask[:, None]))
    return torch.stack(terms).mean() if terms else query.sum() * 0


def path_loss(query, key, paths, path_valid, seeds, anchors, temperature, sigma=0.65):
    """Soft-grid supervision uses beam paths, including sub-cell positions."""
    grid = round(key.shape[2] ** 0.5)
    axis = torch.arange(grid, device=query.device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    coordinates = torch.stack((xx, yy), -1).reshape(-1, 2).float()
    terms = []
    for ai, anchor in enumerate(anchors.tolist()):
        source = query[:, anchor].gather(1, seeds[:, ai, :, None].expand(-1, -1, query.shape[-1]))
        for time in range(anchor + 1, min(anchor + 3, query.shape[1])):
            logits = match_logits(source, key[:, time], temperature)
            distance = (paths[:, ai, :, time, None] - coordinates).square().sum(-1)
            target = (-distance / (2 * sigma**2)).softmax(-1)
            error = -(target * logits.log_softmax(-1)).sum(-1)
            terms.append(masked_mean(error, path_valid[:, ai, :, time]))
    return torch.stack(terms).mean() if terms else query.sum() * 0


def word_spatial_loss(prediction, teacher, content_mask, valid):
    """KL between teacher and student spatial maps of each content word.

    Each content token's attention ``[B, T, cells]`` is normalized over cells, so a
    blurred word map is penalized even when that word carries little of the per-cell
    word distribution.  ``content_mask`` ``[B, L]`` comes from the tokenizer and stop
    words only.
    """
    columns = content_mask.any(0).nonzero().flatten()
    if not len(columns):
        raise ValueError("No content tokens for the word spatial loss")
    content = content_mask[:, columns].float()

    def spatial(attention):
        attention = attention[..., columns].float()
        return attention / attention.sum(-2, keepdim=True).clamp_min(1e-12)

    truth, pred = spatial(teacher), spatial(prediction)
    error = (truth * (truth.clamp_min(1e-12).log() - pred.clamp_min(1e-12).log())).sum(-2)
    weight = content[:, None] * valid[..., None].float()
    return (error * weight).sum() / weight.sum().clamp_min(1)


def depth_smooth(depth, seg_k, valid):
    grid = round(depth.shape[-1] ** 0.5)
    value = depth.reshape(*depth.shape[:2], grid, grid)
    segmentation = (
        F.normalize(seg_k.float(), dim=-1).reshape(*depth.shape[:2], grid, grid, -1).detach()
    )
    terms = []
    for axis in (2, 3):
        a, b = [slice(None)] * 5, [slice(None)] * 5
        a[axis], b[axis] = slice(1, None), slice(None, -1)
        affinity = (segmentation[tuple(a)] * segmentation[tuple(b)]).sum(-1)
        weight = ((affinity - 0.9) / 0.1).clamp(0, 1)
        delta = value[tuple(a[:4])] - value[tuple(b[:4])]
        terms.append(masked_mean(weight * delta.square(), valid))
    return torch.stack(terms).mean()


def structured_loss(
    output,
    target,
    *,
    task,
    samples=64,
    temperature=0.07,
    future_weight=1.0,
    trajectory_weight=0.25,
    cycle_weight=0.01,
    depth_spatial_weight=0.01,
    head_weights=None,
    relation_sampling=None,
    semantic_spatial_weight=0.0,
    content_mask=None,
    depth_sample_weights=None,
    relation_weight=1.0,
    trajectory_sigma=0.65,
    tracking_target="relation",
):
    """``relation_sampling`` (None = uniform) is ``{"seed_fraction", "moving_weight",
    "min_shift"}`` for seed-stratified tracking relation queries.
    ``semantic_spatial_weight`` > 0 adds ``word_spatial_loss`` and needs ``content_mask``
    ``[B, L]``.  ``depth_sample_weights`` ``[B]`` scales each sample's depth loss."""
    if task == "shared":
        weights = dict.fromkeys(HEAD_TASKS, 1.0) if head_weights is None else head_weights
        details, total = {}, None
        for name in HEAD_TASKS:
            loss, values = structured_loss(
                head_output(output, name),
                head_output(target, name),
                task=name,
                samples=samples,
                temperature=temperature,
                future_weight=future_weight,
                trajectory_weight=trajectory_weight,
                cycle_weight=cycle_weight,
                depth_spatial_weight=depth_spatial_weight,
                relation_sampling=relation_sampling,
                semantic_spatial_weight=semantic_spatial_weight,
                content_mask=content_mask,
                depth_sample_weights=depth_sample_weights,
                relation_weight=relation_weight,
                trajectory_sigma=trajectory_sigma,
                tracking_target=tracking_target,
            )
            details.update({f"{name}/{key}": value for key, value in values.items()})
            weighted = weights[name] * loss
            details[f"{name}/weighted"] = weighted.detach()
            total = weighted if total is None else total + weighted
        details["total"] = total.detach()
        return total, details
    details, total = {}, None
    for branch in ("observed", "future"):
        pred, truth = output[branch], target[branch]
        valid = output["timeline"][branch]["valid"][None].expand(
            next(iter(pred.values())).shape[0], -1
        )
        if task.startswith("seg_"):
            error = 1 - F.cosine_similarity(pred["seg_k"].float(), truth["seg_k"].float(), dim=-1)
            loss = masked_mean(error, valid)
        elif task == "depth":
            teacher = (truth["depth"].float() - 4.34375) / 1.65625
            error = F.smooth_l1_loss(pred["depth"].float(), teacher, reduction="none")
            if depth_sample_weights is None:
                loss = masked_mean(error, valid)
            else:
                weight = valid.float() * depth_sample_weights.float()[:, None]
                loss = (error * weight[..., None]).sum() / (
                    weight.sum() * error.shape[-1]
                ).clamp_min(1)
                valid = valid & (depth_sample_weights[:, None] > 0)
            smooth = depth_smooth(pred["depth"].float(), truth["seg_k"], valid)
            loss = loss + depth_spatial_weight * smooth
            details[f"{branch}_spatial"] = smooth.detach()
        elif task == "semantic":
            teacher = truth["text_attention"].float().detach()
            error = F.kl_div(
                pred["text_attention"].float().clamp_min(1e-7).log(), teacher, reduction="none"
            ).sum(-1)
            loss = masked_mean(error, valid)
            if semantic_spatial_weight:
                if content_mask is None:
                    raise ValueError("semantic_spatial_weight requires content_mask")
                spatial = word_spatial_loss(
                    pred["text_attention"], teacher, content_mask.to(teacher.device), valid
                )
                loss = loss + semantic_spatial_weight * spatial
                details[f"{branch}_word_spatial"] = spatial.detach()
        else:
            seeds = None
            if relation_sampling is not None:
                seeds = seed_cells(
                    truth["seeds"],
                    truth["paths"],
                    truth["path_valid"],
                    output["timeline"][branch]["anchor_native"],
                    pred["track_q"].shape[2],
                    relation_sampling["min_shift"],
                )
                details[f"{branch}_moving_seed_cells"] = seeds[1].float().sum(-1).mean()
            if tracking_target == "descriptor":
                # HEFT-style readout tracks by matching descriptors with themselves
                # (K-K), so the student copies the teacher's Q and K vectors directly.
                loss = sum(
                    masked_mean(
                        1 - F.cosine_similarity(pred[key].float(), truth[key].float(), dim=-1),
                        valid,
                    )
                    for key in ("track_q", "track_k")
                )
                details[f"{branch}_descriptor"] = loss.detach()
            elif relation_weight:
                loss = relation_loss(
                    pred["track_q"],
                    pred["track_k"],
                    truth["track_q"],
                    truth["track_k"],
                    valid,
                    samples=samples,
                    temperature=temperature,
                    sampling=relation_sampling,
                    seeds=seeds,
                )
                if relation_weight != 1.0:
                    details[f"{branch}_relation"] = loss.detach()
                    loss = relation_weight * loss
            else:  # foreground beam-track distillation only (path + cycle terms)
                loss = pred["track_q"].sum() * 0
            if trajectory_weight:
                trajectory = path_loss(
                    pred["track_q"],
                    pred["track_k"],
                    truth["paths"],
                    truth["path_valid"],
                    truth["seeds"],
                    output["timeline"][branch]["anchor_native"],
                    temperature,
                    trajectory_sigma,
                )
                loss = loss + trajectory_weight * trajectory
                details[f"{branch}_path"] = trajectory.detach()
            if cycle_weight:
                cycle = cycle_loss(
                    pred["track_q"],
                    pred["track_k"],
                    truth["track_q"],
                    truth["track_k"],
                    valid,
                    samples=samples,
                    temperature=temperature,
                )
                loss = loss + cycle_weight * cycle
                details[f"{branch}_cycle"] = cycle.detach()
        details[branch] = loss.detach()
        weighted = loss * (future_weight if branch == "future" else 1.0)
        total = weighted if total is None else total + weighted
    details["total"] = total.detach()
    return total, details


def augmentation_loss(left, right, task, head_weights=None):
    if task == "shared":
        weights = dict.fromkeys(HEAD_TASKS, 1.0) if head_weights is None else head_weights
        return sum(
            weights[name]
            * augmentation_loss(head_output(left, name), head_output(right, name), name)
            for name in HEAD_TASKS
        )
    losses = []
    for branch in ("observed", "future"):
        valid = left["timeline"][branch]["valid"][None]
        a, b = left[branch], right[branch]
        if task == "tracking":
            for time in range(a["track_q"].shape[1] - 1):
                if not (valid[0, time] and valid[0, time + 1]):
                    continue
                ids = torch.randperm(a["track_q"].shape[2], device=valid.device)[:32]
                x = match_logits(a["track_q"][:, time, ids], a["track_k"][:, time + 1], 0.07)
                y = match_logits(b["track_q"][:, time, ids], b["track_k"][:, time + 1], 0.07)
                losses.append(F.mse_loss(x, y))
        else:
            key = (
                "seg_k"
                if task.startswith("seg_")
                else "depth"
                if task == "depth"
                else "text_attention"
            )
            losses.append(masked_mean((a[key].float() - b[key].float()).square(), valid))
    return torch.stack(losses).mean()
