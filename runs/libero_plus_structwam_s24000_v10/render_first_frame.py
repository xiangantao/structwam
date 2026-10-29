"""First policy frame (after settling, rotate_180 as the OpenWAM client does) of one LIBERO task.

Used to compare image orientation between the LIBERO env that gave 96% on LIBERO-10 (hf_libero) and
LIBERO-plus.  ``python render_first_frame.py SUITE TASK_ID OUT.png``
"""

import sys

import numpy as np
from PIL import Image

suite, task_id, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

task_suite = benchmark.get_benchmark_dict()[suite]()
task = task_suite.get_task(task_id)
bddl = f"{get_libero_path('bddl_files')}/{task.problem_folder}/{task.bddl_file}"
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(0)
env.reset()
obs = env.set_init_state(task_suite.get_task_init_states(task_id)[0])
for _ in range(30):
    obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
head = np.asarray(obs["agentview_image"])[::-1, ::-1]
wrist = np.asarray(obs["robot0_eye_in_hand_image"])[::-1, ::-1]
Image.fromarray(np.concatenate([head, wrist], 1).astype(np.uint8)).save(out)
print(task.language)
env.close()
