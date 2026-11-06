"""(openwam-npu env, one NPU) Encode every eval prompt with the checkpoint's own Reason1 + crossattn_proj,
exactly as the deploy server would, and save {prompt: post-projection context (CPU bf16)}.

    npu_entry.py precompute_prompt_cache.py CKPT_DIR CKPT_NAME PROMPTS_JSON OUT_PT

The server loads it through OPENWAM_PROMPT_CACHE_FILE (openwam/deploy/engine.py). The result depends on
the checkpoint (its crossattn_proj), so build it once per checkpoint.
"""
import json
import os
import sys
import time

import torch

os.environ.pop("OPENWAM_TEXT_ENCODER_DEVICE", None)  # encode on the NPU, like the server's original path
from openwam.deploy.model_loader import load_from_checkpoint_dir  # noqa: E402

ckpt_dir, ckpt_name, prompts_json, out = sys.argv[1:5]
prompts = json.load(open(prompts_json))
_, arch = load_from_checkpoint_dir(ckpt_dir, device="cuda", ckpt_name=ckpt_name)
backbone = arch.video_backbone
cache, start = {}, time.time()
with torch.no_grad():
    for i, prompt in enumerate(prompts):
        context = backbone._encode_text_context(prompt, device=torch.device("cuda"), dtype=torch.bfloat16)
        cache[prompt] = context.detach().to("cpu")
        if i % 100 == 0:
            print(f"{i}/{len(prompts)} {time.time() - start:.0f}s shape {tuple(context.shape)}", flush=True)
torch.save(cache, out)
print(f"saved {len(cache)} prompts -> {out}", flush=True)
