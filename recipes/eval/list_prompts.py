"""(LIBERO-plus env) Every instruction the eval client sends: clean(task.language) over all 4 suites.

    client_python.sh list_prompts.py OUT_JSON

10030 tasks give 1549 distinct prompts (most of them Language-perturbation rewrites).
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ["STRUCTWAM_REPO"])
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "StructWAM" / "benchmarks" / "libero-plus"))
sys.path.insert(0, str(REPO / "StructWAM"))
from clean_instructions import clean  # noqa: E402
from single_eval import _write_libero_config  # noqa: E402

_write_libero_config()
from libero.libero import benchmark  # noqa: E402

BASE = json.loads((HERE / "libero_base_instructions.json").read_text())
out = sys.argv[1]
prompts, n = set(), 0
suites = benchmark.get_benchmark_dict()
for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
    task_suite = suites[suite]()
    for i in range(task_suite.n_tasks):
        prompts.add(clean(task_suite.get_task(i).language, BASE))
        n += 1
json.dump(sorted(prompts), open(out, "w"), indent=0)
print(f"{n} tasks -> {len(prompts)} distinct prompts -> {out}")
