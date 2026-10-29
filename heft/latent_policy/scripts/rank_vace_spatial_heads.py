#!/usr/bin/env python3
"""Rank cached no-prompt attn1 heads by same-frame object affinity."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score


def surface_token_labels(source, times, grid=32):
    names = source['point_label'].astype(str)
    names = np.asarray(['gripper' if name in {'gripper base', 'left finger', 'right finger'}
                        else name for name in names])
    unique = np.unique(names)
    class_ids = np.searchsorted(unique, names)
    labels = np.full((len(times), grid * grid), -1, np.int64)
    for ti, frame in enumerate(times):
        visible = source['visible'][frame] & np.isfinite(source['xy'][frame]).all(-1)
        pixels = source['xy'][frame, visible]
        cells = np.floor((pixels + .5) * grid / 256).astype(int).clip(0, grid - 1)
        flat = cells[:, 1] * grid + cells[:, 0]
        occupied = np.zeros((len(unique), grid * grid), bool)
        occupied[class_ids[visible], flat] = True
        # Mixed-object cells and cells without visible surface samples are unknown.
        clean = occupied.sum(0) == 1
        labels[ti, clean] = occupied[:, clean].argmax(0)
    return labels, unique


def score_head(q, k, labels, names, device):
    by_object = {}
    for ti in range(len(labels)):
        cells = np.flatnonzero(labels[ti] >= 0)
        ids = labels[ti, cells]
        queries = torch.as_tensor(q[ti, cells].astype(np.float32), device=device)
        keys = torch.as_tensor(k[ti, cells].astype(np.float32), device=device)
        affinity = (queries @ keys.T / q.shape[-1] ** .5).cpu().numpy()
        for obj in np.unique(ids):
            rows = np.flatnonzero(ids == obj)
            if len(rows) < 2 or len(rows) == len(cells):
                continue
            values = []
            for row in rows:
                valid = np.arange(len(cells)) != row
                positive = (ids[valid] == obj).astype(int)
                scores = affinity[row, valid]
                values.append({
                    'ap': float(average_precision_score(positive, scores)),
                    'chance': float(positive.mean()),
                    'top1_same_object': float(positive[scores.argmax()]),
                })
            by_object.setdefault(str(names[obj]), []).extend(values)
    per_object = {}
    for name, values in by_object.items():
        per_object[name] = {key: float(np.mean([row[key] for row in values]))
                            for key in ('ap', 'chance', 'top1_same_object')}
        per_object[name]['ap_lift'] = per_object[name]['ap'] - per_object[name]['chance']
    return {'per_object': per_object,
            **{key: float(np.mean([row[key] for row in per_object.values()]))
               for key in ('ap', 'chance', 'top1_same_object', 'ap_lift')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--tracks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    with np.load(args.tracks) as z:
        source = {key: z[key] for key in ('point_label', 'visible', 'xy')}
    rows = []
    with np.load(args.cache) as z:
        labels, names = surface_token_labels(source, z['frame_indices'])
        for key in sorted(z.files):
            if not key.endswith('_q'):
                continue
            layer = int(key.split('_')[1])
            q, k = z[key], z[key[:-1] + 'k']
            for head in range(q.shape[0]):
                row = {'layer': layer, 'head': head,
                       **score_head(q[head], k[head], labels, names, device)}
                rows.append(row)
            print(json.dumps({'layer_scored': layer}), flush=True)
        frames = z['frame_indices'].tolist()
    rows.sort(key=lambda row: -row['ap_lift'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'protocol': 'empty-prompt attn1; same-frame Q/K; exclude diagonal; equal object weight; unknown labels ignored',
              'cache': str(args.cache), 'tracks': str(args.tracks),
              'frame_indices': frames, 'ranking': rows}
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({'output': str(args.output), 'top': [{key: row[key] for key in
                      ('layer', 'head', 'ap', 'ap_lift', 'top1_same_object')} for row in rows[:12]]}), flush=True)


if __name__ == '__main__':
    main()
