"""Run benchmarks/libero-plus/single_eval.py with the perturbation ids stripped from the instruction.

LIBERO-plus derives ``Task.language`` from the bddl file name, so every non-Language task carries
its perturbation ids ("... table 18", "... view 0 0 100 2 352 initstate 0 noise 3").  The policy
gets the original LIBERO instruction instead (``clean_instructions.clean``); Language
Instructions tasks are unchanged.  Everything else in the client is untouched.

    python clean_prompt_entry.py /path/to/single_eval.py [single_eval args]
"""

import json
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import openwam2libero_interface as interface  # noqa: E402  (on PYTHONPATH via the scheduler)
from clean_instructions import clean  # noqa: E402

BASE = json.loads((HERE / "libero_base_instructions.json").read_text())
_act = interface.OpenWAMLiberoPolicy.act


def _act_clean(self, obs, prompt):
    sent = clean(prompt, BASE)
    if getattr(self, "_logged_prompt", None) != prompt:
        print(f"[clean-prompt] {prompt!r} -> {sent!r}", flush=True)
        self._logged_prompt = prompt
    return _act(self, obs, sent)


interface.OpenWAMLiberoPolicy.act = _act_clean

if __name__ == "__main__":
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")
