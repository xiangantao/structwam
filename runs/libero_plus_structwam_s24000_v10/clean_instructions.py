"""LIBERO-plus task.language -> the instruction the policy should see.

LIBERO-plus builds ``Task.language`` from the bddl file name for every category except Language
Instructions, so the perturbation ids leak into the instruction ("... on the plate table 18",
"... view 0 0 100 2 352 initstate 0 noise 3", "... add 10", "... light 1").  The clean instruction
is the original LIBERO one the name starts with (longest match on a word boundary), and only
when everything after it is id tokens (``ID_WORDS`` and numbers): Language Instructions tasks,
some of which append real distractor clauses to the original instruction, are left as they are.

    python clean_instructions.py build BASE_JSON    # in the hf_libero env: the 40 LIBERO instructions
    python clean_instructions.py check BASE_JSON    # in the LIBERO-plus env: map all 10,030 tasks
"""

import json
import re
import sys

ID_WORDS = {"table", "tb", "view", "initstate", "noise", "add", "light", "moved"}
ID_TOKEN = re.compile(r"(level|sample)\d+")


def is_id_suffix(rest: str) -> bool:
    tokens = rest.split()
    numbered = [t for t in tokens if t.isdigit() or ID_TOKEN.fullmatch(t)]
    return bool(numbered) and all(t in ID_WORDS or t in numbered for t in tokens)


def clean(language: str, base: list) -> str:
    matches = [b for b in base if language.startswith(b + " ")]
    if not matches:
        return language
    original = max(matches, key=len)
    return original if is_id_suffix(language[len(original):]) else language


if __name__ == "__main__":
    mode, path = sys.argv[1], sys.argv[2]
    from libero.libero import benchmark

    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    if mode == "build":
        instructions = sorted({benchmark.get_benchmark_dict()[s]().get_task(i).language
                               for s in suites for i in range(10)})
        json.dump(instructions, open(path, "w"), indent=1)
        print(f"{len(instructions)} base instructions -> {path}")
    else:
        import collections
        import os

        base = json.load(open(path))
        cls = json.load(open(os.path.join(os.environ["LIBERO_PLUS_PATH"], "libero/libero/benchmark/task_classification.json")))
        counts = collections.defaultdict(lambda: [0, 0, 0])   # cleaned, unchanged, total
        examples = {}
        leftover = collections.Counter()   # suffix words of non-Language tasks that were not cleaned
        bd = benchmark.get_benchmark_dict()
        for suite, records in cls.items():
            tasks = bd[suite]()
            for item in records:
                language = tasks.get_task(int(item["id"]) - 1).language
                cleaned = clean(language, base)
                row = counts[item["category"]]
                row[0 if cleaned != language else 1] += 1
                row[2] += 1
                examples.setdefault(item["category"], (language, cleaned))
                if cleaned == language and item["category"] != "Language Instructions":
                    leftover[language] += 1
        for category, (cleaned, unchanged, total) in counts.items():
            print(f"{category:22s} cleaned {cleaned:5d}  unchanged {unchanged:5d}  of {total}")
            print(f"    {examples[category][0]!r}\n -> {examples[category][1]!r}")
        print(f"non-Language tasks left unchanged: {sum(leftover.values())} {list(leftover)[:5]}")
