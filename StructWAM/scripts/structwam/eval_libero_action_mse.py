"""Offline generated-action MSE; never passes future targets to generation."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from openwam.deploy.model_loader import load_from_checkpoint_dir
from openwam.deploy.denoise_schedule import make_schedule
from openwam.dataloader import build_dataset

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--per-task', type=int, default=2)
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--seed', type=int, default=20261021)
    p.add_argument('--manifest', default=None)
    p.add_argument('--structure-video-steps', type=int, default=None)
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    ckpt = Path(args.checkpoint)
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(ckpt.parent / 'config.yaml')
    ds = build_dataset(cfg.dataloader, split='train')
    groups = {}
    for ep, row in ds._eps_df.iterrows():
        length = int(ds._ep_valid_end[ep] - ds._ep_valid_start[ep])
        if length < 33:
            continue
        table = ds._load_data_table(int(row['data/chunk_index']), int(row['data/file_index']))
        win = table.slice(int(ds._ep_data_row_offset[ep]) + int(ds._ep_valid_start[ep]), 1).to_pandas()
        prompt = ds._resolve_prompt(row, win)
        groups.setdefault(prompt, []).append(ep)
    rng = np.random.default_rng(args.seed)
    selected = []
    for prompt in sorted(groups):
        for ep in rng.choice(groups[prompt], size=args.per_task, replace=False):
            ep = int(ep)
            available = int(ds._ep_valid_end[ep] - ds._ep_valid_start[ep]) - 33
            offset = int(rng.integers(0, available + 1))
            index = int(ds._cum_n_starts[ep]) + offset // ds._window_stride
            selected.append(dict(index=index, episode=int(ds._eps_df.iloc[ep]['episode_index']), offset=offset, prompt=prompt))
    if args.manifest:
        reference = json.loads(Path(args.manifest).read_text())
        assert selected == reference['selected'], 'Sample selection differs from reference'
    metadata = dict(checkpoint=str(ckpt), checkpoint_bytes=ckpt.stat().st_size,
        split='train (NOT held out)', samples=len(selected), tasks=len(groups),
        denoise_steps=args.steps, schedule='sync', cfg_scale=1.0, seed=args.seed,
        action_horizon=32, metric='generated normalized EEF10 MSE; mask excludes inactive unified dims and padded timesteps',
        conditioning='current image canvas + current normalized proprio + prompt; no future images/actions',
        selected=selected, structure_video_steps=args.structure_video_steps)
    if str(cfg.dataloader.type) == 'libero_struct':
        metadata['conditioning'] += '; structure uses past/current 33 frames only; no phase labels supplied'
    (out/'manifest.json').write_text(json.dumps(metadata, indent=2))
    print('MANIFEST',json.dumps({k:v for k,v in metadata.items() if k!='selected'}), flush=True)
    cfg, arch = load_from_checkpoint_dir(str(ckpt.parent), device='cuda:0', ckpt_name=ckpt.name)
    arch.requires_grad_(False)
    if args.structure_video_steps is not None:
        arch.inference_cfg['video_steps_near'] = args.structure_video_steps
        arch.inference_cfg['video_steps_far'] = args.structure_video_steps
    # Return model-space actions; preserve the saved normalizer for an additional raw-EEF10 metric.
    normalizer = arch.normalizer
    arch.normalizer = None
    cache = {}
    totals = {k:[0.0,0] for k in ['mse','xyz','rot6d','gripper','first_action','zero_normalized_baseline','raw_eef10']}
    per_sample = []
    for i, item in enumerate(selected):
        started = time.monotonic()
        sample = ds._getitem_impl(item['index'])  # no silent fallback to a different sample
        assert sample['prompt'] == item['prompt']
        target = sample['action'].float().numpy()
        mask = sample['action_mask'].numpy().astype(bool)
        assert mask.sum() == 320 and target.shape == (32,80)
        schedule = make_schedule('sync', video_scheduler=arch.video_scheduler,
            action_scheduler=arch.action_scheduler, num_steps=args.steps,
            shift=arch.action_backbone.shift_action, shift_video=arch.video_backbone.shift_video)
        with torch.inference_mode():
            extra = {'structure_frames':sample['structure_frames']} if 'structure_frames' in sample else {}
            result = arch.generate(schedule, sample['prompt'], first_frame_image=sample['first_frame_image'],
                num_frames=len(sample['video']), action_num_frames=33,
                height=int(cfg.dataloader.height), width=int(cfg.dataloader.width),
                seed=args.seed+i, num_inference_steps=args.steps, shift=arch.action_backbone.shift_action,
                decode_video=False, proprio=sample['proprio'].to(device=arch.device,dtype=arch.dtype),
                active_action_mask=torch.from_numpy(mask.any(0)), prompt_embed_cache=cache, cfg_scale=1.0, **extra)
        pred = np.asarray(result['actions'], dtype=np.float64)
        assert pred.shape == target.shape and np.isfinite(pred).all()
        err = (pred-target)**2
        raw_pred = normalizer.unnormalize(pred.copy())
        raw_target = normalizer.unnormalize(target.copy())
        metrics = {}
        for key, values in [('mse',err[mask]), ('xyz',err[:,:3][mask[:,:3]]),
            ('rot6d',err[:,3:9][mask[:,3:9]]), ('gripper',err[:,9][mask[:,9]]),
            ('first_action',err[0][mask[0]]), ('zero_normalized_baseline',target[mask]**2),
            ('raw_eef10',(raw_pred-raw_target)**2)]:
            totals[key][0] += float(values.sum()); totals[key][1] += int(values.size)
            metrics[key] = float(values.mean())
        record = dict(sample=i+1, **item, **metrics, seconds=time.monotonic()-started)
        for key in ('structure_gate', 'video_steps'):
            if key in result: record[key] = result[key]
        per_sample.append(record)
        with (out/'samples.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        np.savez_compressed(out/f'sample_{i:03d}.npz', predicted=pred,target=target,mask=mask)
        print('EVAL', json.dumps(record), flush=True)
    summary = {k:v[0]/v[1] for k,v in totals.items()}
    summary.update(samples=len(selected),tasks=len(groups),denoise_steps=args.steps,split=metadata['split'],
        checkpoint=str(ckpt),action_rmse=totals['mse'][0]**.5/totals['mse'][1]**.5,
        per_task={p:float(np.mean([r['mse'] for r in per_sample if r['prompt']==p])) for p in sorted(groups)})
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('COMPLETE',json.dumps(summary),flush=True)

if __name__ == '__main__':
    main()
