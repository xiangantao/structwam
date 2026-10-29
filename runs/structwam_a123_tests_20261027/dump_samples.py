# Dump libero_struct samples (aug off, and the b2 aug config with a fixed seed) for before/after comparison.
import random, sys, numpy as np
from omegaconf import OmegaConf
from openwam.dataloader import build_dataset
cfg = OmegaConf.load('/home/ma-user/work/2337888765/workspace/runs/structwam_step40000_eval_ckpt/config.yaml').dataloader
out = {}
for tag, c in (('off', cfg), ('aug', OmegaConf.merge(cfg, {'structure_augment': {'enabled': True}}))):
    ds = build_dataset(c, split='train')
    for i in (1234, 98765):
        random.seed(7); s = ds[i]
        out[f'{tag}_{i}_video'] = np.stack([np.asarray(f) for f in s['video']])
        for cam, clip in s['structure_frames'].items():
            out[f'{tag}_{i}_hist_{cam}'] = clip
        out[f'{tag}_{i}_action'] = np.asarray(s['action'])
np.savez_compressed(sys.argv[1], **out)
print('dumped', len(out))
