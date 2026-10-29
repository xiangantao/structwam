# B2 data path throughput: libero_struct with structure_augment on, batch 8, 4 workers (one rank's loader).
import time
import torch
from omegaconf import OmegaConf
from openwam.dataloader import build_dataset
cfg = OmegaConf.load('/home/ma-user/work/2337888765/workspace/runs/structwam_step40000_eval_ckpt/config.yaml').dataloader
cfg = OmegaConf.merge(cfg, {'structure_augment': {'enabled': True}})
ds = build_dataset(cfg, split='train')
dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, num_workers=4, collate_fn=list, persistent_workers=True)
t0 = time.time(); times = []
for i, batch in enumerate(dl):
    times.append(time.time()); 
    if i == 24: break
print('first batch %.1fs' % (times[0] - t0))
steady = (times[-1] - times[4]) / 20
print('steady %.2f s/batch over 20 batches (training step is ~8.6 s)' % steady)
print('keys', sorted(batch[0].keys())[:12], 'head hist', batch[0]['structure_frames']['head'].shape)
