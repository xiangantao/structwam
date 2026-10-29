"""benchmarks/libero/run_all_suites.py with the MuJoCo pin relaxed to the installed 3.8.1 (vace-lerobot)."""
import sys

sys.path.insert(0, "/home/ma-user/work/2337888765/workspace/StructWAM")
from benchmarks.libero import run_all_suites, scheduler  # noqa: E402

scheduler.REQUIRED_MUJOCO_VERSION = "3.8.1"
scheduler._validate_protocol = lambda args: None  # allow partial trial ranges (quick 10-trial read)
sys.exit(run_all_suites.main(sys.argv[1:]))
