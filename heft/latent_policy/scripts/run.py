"""Register the local plugin and dispatch to the official LeRobot train/eval runners."""

import os
import runpy
import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "eval", "prepare"):
        raise SystemExit(
            "Usage: run.py {train|eval|prepare} [official runner/config preparation arguments]"
        )
    mode = sys.argv.pop(1)
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[name] = "1"
    os.environ["WANDB_MODE"] = "offline"
    import lerobot_policy_latent_act  # noqa: F401

    if mode == "prepare":
        from lerobot_policy_latent_act.data import main as prepare

        prepare()
    else:
        runpy.run_module(f"lerobot.scripts.lerobot_{mode}", run_name="__main__")


if __name__ == "__main__":
    main()
