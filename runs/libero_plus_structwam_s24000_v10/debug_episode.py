"""One LIBERO-plus episode through the canonical client, with the policy inputs saved as images.

Runs benchmarks/libero-plus/single_eval.py unchanged (official one-rollout protocol), plus: every
``--every`` steps the head / wrist frames exactly as sent to the policy (after rotate_180) are saved
next to debug mode's per-step meta.json.  ``--prompt`` replaces the task's instruction (ablation).

    python debug_episode.py --suite libero_spatial --task-id 8 --port 9150 --out DIR [--prompt "..."]
"""

import argparse
import sys
import tempfile
from pathlib import Path

import yaml
from PIL import Image

BENCH = Path("/home/ma-user/work/2337888765/workspace/StructWAM/benchmarks/libero-plus")
sys.path.insert(0, str(BENCH))

import openwam2libero_interface as interface  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--suite", required=True)
parser.add_argument("--task-id", type=int, required=True)
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--every", type=int, default=5)
parser.add_argument("--prompt")
args = parser.parse_args()

_save_debug = interface.OpenWAMLiberoPolicy._maybe_debug
_act = interface.OpenWAMLiberoPolicy.act


def _maybe_debug(self, obs, payload, model_action, action):
    _save_debug(self, obs, payload, model_action, action)
    if self._step % args.every == 0:
        step_dir = self._debug_dir / f"episode_{self._episode:03d}" / f"step_{self._step:04d}"
        Image.fromarray(self._image(obs, self._head_camera_key)).save(step_dir / "head.png")
        Image.fromarray(self._image(obs, self._left_wrist_camera_key)).save(step_dir / "wrist.png")


def _act_with_prompt(self, obs, prompt):
    if not getattr(self, "_logged_prompt", False):
        (self._debug_dir / "prompt.txt").write_text(f"task language: {prompt}\nsent: {args.prompt or prompt}\n")
        self._logged_prompt = True
    return _act(self, obs, args.prompt or prompt)


interface.OpenWAMLiberoPolicy._maybe_debug = _maybe_debug
interface.OpenWAMLiberoPolicy.act = _act_with_prompt

import single_eval  # noqa: E402  (imports the patched class)

args.out.mkdir(parents=True, exist_ok=True)
config = yaml.safe_load((BENCH / "policy_config.yml").read_text())
config.update(debug=True, debug_dir=str(args.out / "frames"))
config_path = Path(tempfile.mkdtemp()) / "policy_config_debug.yml"
config_path.write_text(yaml.safe_dump(config))
sys.exit(single_eval.main(["--config", str(config_path), "--suite", args.suite, "--task-id", str(args.task_id),
                           "--host", "127.0.0.1", "--port", str(args.port), "--trial-start", "0",
                           "--num-trials", "1", "--result-dir", str(args.out / "result")]))
