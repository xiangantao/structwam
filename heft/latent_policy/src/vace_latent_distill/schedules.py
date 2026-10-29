"""Sparse UniPC updates that retain the original future readout endpoint."""

import numpy as np
import torch


def sparse_schedule(template, config, updates=None, device=None):
    from diffusers import UniPCMultistepScheduler

    updates = config.denoise_updates if updates is None else updates
    maximum = config.denoise_reference_updates
    if not 1 <= updates <= maximum:
        raise ValueError("Sparse update count must be between 1 and the reference count")
    scheduler = UniPCMultistepScheduler.from_config(template.config)
    scheduler.set_timesteps(config.scheduler_steps, device=device)
    begin = int((scheduler.sigmas[:-1] - config.sigma).abs().argmin())
    endpoint = begin + maximum
    if endpoint >= len(scheduler.timesteps):
        raise ValueError("The reference endpoint must still have a capture timestep")
    relative = np.rint(np.linspace(0, maximum, updates + 1)).astype(int)
    # Keep prefix/suffix nodes: the readout endpoint is not a terminal solver step.
    indices = torch.cat((
        torch.arange(begin),
        torch.from_numpy(begin + relative[:-1]),
        torch.arange(endpoint, len(scheduler.timesteps)),
    ))
    scheduler.sigmas = torch.cat((scheduler.sigmas[indices], scheduler.sigmas[-1:]))
    scheduler.timesteps = scheduler.timesteps[indices.to(scheduler.timesteps.device)]
    scheduler.num_inference_steps = len(scheduler.timesteps)
    scheduler.set_begin_index(begin)
    nodes = [
        {"node": node, "reference_step": int(reference_step),
         "sigma": float(scheduler.sigmas[begin + node]),
         "timestep": float(scheduler.timesteps[begin + node])}
        for node, reference_step in enumerate(relative)
    ]
    return scheduler, begin, nodes
