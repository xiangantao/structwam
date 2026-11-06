"""(openwam-npu env, one NPU) UMT5 embeddings for every eval prompt missing from the structure student's
text cache, computed exactly like latent_dp.online.OnlineMaps._encode, saved as bf16 (lossless: the
values are bf16 hidden states times a 0/1 mask) for mmap loading by the eval servers.

    npu_entry.py precompute_student_text.py STUDENT_CKPT TEXT_ROOT PROMPTS_JSON OUT_PT

TEXT_ROOT is the UMT5 bundle (Wan2.1-T2V-1.3B-Diffusers: text_encoder/ + tokenizer/). The servers read
the result through OPENWAM_STUDENT_TEXT_EXTRA (heft/latent_policy/src/latent_dp/online.py), so they never
load the 11 GB UMT5 encoder onto the NPU for Language-perturbation instructions.
"""
import json
import sys
import time
from pathlib import Path

import torch
from latent_dp.online import _text_cache_path
from transformers import AutoTokenizer, UMT5EncoderModel

student_ckpt, text_root, prompts_json, out = sys.argv[1:5]
saved = torch.load(student_ckpt, map_location="cpu", weights_only=False, mmap=True)
cache = torch.load(_text_cache_path(student_ckpt, saved), map_location="cpu", weights_only=True)
root = Path(text_root)
del saved
prompts = [p for p in json.load(open(prompts_json)) if p not in cache]
print(f"{len(cache)} prompts already in the student cache; encoding {len(prompts)} missing ones", flush=True)
tok = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
enc = UMT5EncoderModel.from_pretrained(root / "text_encoder", torch_dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()
extra, t = {}, time.time()
with torch.no_grad():
    for i, p in enumerate(prompts):
        tokens = tok([p], padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        mask = tokens["attention_mask"].to("cuda")
        hidden = enc(input_ids=tokens["input_ids"].to("cuda"), attention_mask=mask).last_hidden_state
        value = (hidden[0].float() * mask[0, :, None]).cpu()
        packed = value.to(torch.bfloat16)
        assert torch.equal(packed.float(), value), "bf16 packing would not be lossless"
        extra[p] = packed
        if i % 100 == 0:
            print(f"{i}/{len(prompts)} {time.time() - t:.0f}s", flush=True)
torch.save(extra, out)
print(f"saved {len(extra)} prompts -> {out}", flush=True)
