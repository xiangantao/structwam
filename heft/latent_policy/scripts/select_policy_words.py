#!/usr/bin/env python3
"""Fix, per task, the instruction tokens whose relevance the policy features report.

Parser-free: content tokens are grouped into words at the tokenizer's own word boundaries
(a piece without the leading "▁" continues the previous word); a word's map is the sum of
its pieces' teacher maps at the last observed time.  The ``--count`` words whose map is most
spatially concentrated (top-5% mass), averaged over that task's cached training windows
(both cameras), are kept, in instruction order.  The choice is made
once per task so training and rollout features use the same tokens.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OBSERVED_LAST = 11


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    from vace_latent_distill.config import PipelineConfig
    from vace_latent_distill.offline import unpack_target
    from vace_latent_distill.teacher import content_token_masks, local_scripts

    config = PipelineConfig.load(args.cache_dir / "pipeline.json")
    rows = [json.loads(l) for l in (args.cache_dir / "manifest_train.jsonl").read_text().splitlines() if l.strip()]
    tasks = sorted({r["task"] for r in rows})
    content = content_token_masks(config.teacher.text_root, tasks)
    local_scripts()
    tokenizer = AutoTokenizer.from_pretrained(Path(config.teacher.text_root) / "tokenizer", local_files_only=True)
    groups = {}
    for task in tasks:
        pieces = tokenizer.convert_ids_to_tokens(tokenizer(task, padding="max_length", max_length=512,
                                                           truncation=True)["input_ids"])
        words = []
        for index in np.flatnonzero(content[task]):
            if words and not pieces[index].startswith(chr(9601)) and words[-1][-1] == index - 1:
                words[-1].append(int(index))
            else:
                words.append([int(index)])
        groups[task] = (words, pieces)
    scores = defaultdict(list)
    for row in rows:
        fields = unpack_target(torch.load(row["target_path"], map_location="cpu", weights_only=True),
                               identity=row["provenance_id"])
        attention = fields["observed"]["text_attention"][OBSERVED_LAST].float().numpy()   # [cells, L]
        maps = np.stack([attention[:, word].sum(1) for word in groups[row["task"]][0]], 1)
        maps = maps / np.maximum(maps.sum(0, keepdims=True), 1e-8)
        scores[row["task"]].append(np.sort(maps, 0)[-51:].sum(0))                     # top-5% mass
    result = {}
    for task in tasks:
        words, pieces = groups[task]
        mean = np.mean(scores[task], 0)
        chosen = sorted(np.argsort(-mean)[: args.count].tolist())
        result[task] = {"words": [words[i] for i in chosen],
                        "text": ["".join(pieces[j] for j in words[i]).replace(chr(9601), "") for i in chosen],
                        "concentration": [round(float(mean[i]), 3) for i in chosen],
                        "windows": len(scores[task])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    for task, item in result.items():
        print(f"{item['text']}  <- {task}")


if __name__ == "__main__":
    main()
