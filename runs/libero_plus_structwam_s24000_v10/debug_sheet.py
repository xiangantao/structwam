"""Contact sheet + action summary for a debug_episode.py run.

    python debug_sheet.py DEBUG_DIR [STEP ...]
Writes DEBUG_DIR/sheet.png (head | wrist per step, policy inputs after rotate_180) and prints the
end-effector position, gripper state and commanded action every 20 steps.
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

root = Path(sys.argv[1])
frames = sorted((root / "frames").glob("episode_*"))[0]
steps = [int(s) for s in sys.argv[2:]] or [0, 25, 50, 75, 100, 150, 200, 300, 400, 595]
tile = 192
sheet = Image.new("RGB", (4 * tile, ((len(steps) + 1) // 2) * (tile + 16)), "white")
draw = ImageDraw.Draw(sheet)
for index, step in enumerate(steps):
    step_dir = frames / f"step_{step:04d}"
    x, y = (index % 2) * 2 * tile, (index // 2) * (tile + 16)
    for offset, name in enumerate(("head", "wrist")):
        path = step_dir / f"{name}.png"
        if path.exists():
            sheet.paste(Image.open(path).convert("RGB").resize((tile, tile)), (x + offset * tile, y + 16))
    draw.text((x + 4, y + 2), f"step {step}  (head | wrist)", fill="black")
sheet.save(root / "sheet.png")

print(open(root / "frames" / "prompt.txt").read().strip())
print("step | eef xyz (state[0:3]) | gripper state[9] | action7 xyz*100, rot*100, grip")
for step_dir in sorted(frames.glob("step_*")):
    step = int(step_dir.name.split("_")[1])
    if step % 20:
        continue
    meta = json.loads((step_dir / "meta.json").read_text())
    state, action = meta["state"], meta["libero_action7"]
    print(f"{step:4d} | {state[0]:+.3f} {state[1]:+.3f} {state[2]:+.3f} | {state[9]:+.3f} | "
          f"{action[0]*100:+5.1f} {action[1]*100:+5.1f} {action[2]*100:+5.1f}  "
          f"{action[3]*100:+5.1f} {action[4]*100:+5.1f} {action[5]*100:+5.1f}  {action[6]:+.0f}")
