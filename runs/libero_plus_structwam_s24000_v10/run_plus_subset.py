"""benchmarks/libero-plus/run_all_suites.py restricted to some perturbation categories.

The job list is the scheduler's own (``--task-sample-ratio`` / ``--task-sample-seed`` keep the
same 10% sample as earlier runs), filtered to the categories in ``PLUS_CATEGORIES`` (comma
separated, names as in task_classification.json).  Everything else is the canonical launcher.

    PLUS_CATEGORIES="Camera Viewpoints,Sensor Noise" python run_plus_subset.py [run_all_suites args]
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

BENCH = Path("/home/ma-user/work/2337888765/workspace/StructWAM/benchmarks/libero-plus")
REPO = BENCH.parents[1]
sys.path.insert(0, str(REPO))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = _load("libero_plus_run_all_suites", BENCH / "run_all_suites.py")
scheduler = _load("libero_plus_scheduler", BENCH / "scheduler.py")
# The same patches run_all_suites.main applies to its scheduler.
scheduler.SCRIPT_DIR = BENCH
scheduler.DEFAULT_POLICY_CONFIG = BENCH / "policy_config.yml"
scheduler.DEFAULT_OUTPUT_ROOT = REPO / "outputs" / "libero-plus"
scheduler.LIBERO_PROTOCOL_VERSION = "openwam-libero-plus-native-action-seed10000-settle30-v1"
scheduler._client_command = launcher._native_client_command
scheduler._client_env = launcher._native_client_env
scheduler._read_result = launcher._native_read_result
scheduler._websocket_server_is_ready = scheduler._port_is_open

CATEGORIES = {c.strip() for c in os.environ["PLUS_CATEGORIES"].split(",") if c.strip()}
_build_jobs = scheduler._build_jobs


def _build_category_jobs(*args, **kwargs):
    jobs = _build_jobs(*args, **kwargs)
    argv = sys.argv[1:]
    libero_path = Path(argv[argv.index("--libero-path") + 1])
    payload = json.loads((libero_path / "libero/libero/benchmark/task_classification.json").read_text())
    category = {(suite, int(item["id"]) - 1): item["category"] for suite, items in payload.items() for item in items}
    kept = [job for job in jobs if category[(job.suite, job.task_id)] in CATEGORIES]
    print(f"category filter {sorted(CATEGORIES)}: {len(kept)} of {len(jobs)} sampled tasks", flush=True)
    return kept


scheduler._build_jobs = _build_category_jobs

if __name__ == "__main__":
    sys.exit(scheduler.main(sys.argv[1:]))
