#!/usr/bin/env python3
"""Fix, per task, the instruction tokens of every heat channel of the policy maps.

Grammar roles (``latent_dp.policy_maps.instruction_roles``: target / destination of up to two
sub-goals, locator) plus the automatically selected words of ``select_policy_words.py``, and
the content-word tokens used to renormalize the +16 heat.  Only the instruction text is used.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--words", type=Path, required=True, help="select_policy_words.py output (all tasks)")
    parser.add_argument("--cache-dir", type=Path, required=True, help="for pipeline.json (tokenizer path)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    from latent_dp.policy_maps import role_tokens
    from vace_latent_distill.config import PipelineConfig

    text_root = PipelineConfig.load(args.cache_dir / "pipeline.json").teacher.text_root
    tokenizer = AutoTokenizer.from_pretrained(Path(text_root) / "tokenizer", local_files_only=True)
    words = json.loads(args.words.read_text())
    result = {}
    for task in sorted(words):
        tokens, readable = role_tokens(task, tokenizer, words[task])
        result[task] = {"tokens": tokens, "readable": readable,
                        "length": max(t for ts in tokens.values() for t in ts) + 1}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    for task, item in result.items():
        print({k: v for k, v in item["readable"].items() if v}, "<-", task)


if __name__ == "__main__":
    main()
