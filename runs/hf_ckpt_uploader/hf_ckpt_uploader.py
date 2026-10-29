"""Watch training runs and upload every new checkpoint to one private Hugging Face model repo.

Layout in the repo: ``<run>/checkpoint_step_<N>.safetensors`` plus, once per run, the files next
to it (config.yaml, normalization_stats.npy, reason1/) and the run's command.txt.  The frozen
Cosmos-Reason1 and VAE weights are byte-identical in every checkpoint, so with hf_xet only the
trained ~6 GB of each new file is actually transferred.

A checkpoint is uploaded once it has not been modified for ``--settle`` seconds, oldest first
(the trainer's rotation deletes the oldest).  It is first hard-linked into ``--staging`` (same file
system, no copy) so the rotation cannot delete it mid-upload; the link is removed after the
upload.  Progress lives in ``--state`` (JSON), so the watcher can be restarted at any time.
Optimizer states (accel_state_step_*) are never uploaded.

The token comes from the usual huggingface_hub sources (HF_TOKEN, or ``hf auth login`` under the
same HF_HOME); it is never printed.
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

STEP = re.compile(r"checkpoint_step_(\d+)\.safetensors$")
RUN_FILES = ("config.yaml", "normalization_stats.npy")


def log(msg):
    print(time.strftime("%m-%d %H:%M:%S"), msg, flush=True)


def load_state(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {"uploaded": {}, "run_files": []}


def save_state(path, state):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True))
    os.replace(tmp, path)


def discover(roots, extras, every):
    """[(run, step, path, run_dir)] for every checkpoint on disk with step % every == 0."""
    found = []
    for root in roots:
        for path in glob.glob(os.path.join(root, "*", "checkpoints", "*", "checkpoint_step_*.safetensors")):
            run_dir = Path(path).parents[2]
            found.append((run_dir.name, int(STEP.search(path)[1]), path, run_dir))
    for run, directory in extras:
        for path in glob.glob(os.path.join(directory, "checkpoint_step_*.safetensors")):
            found.append((run, int(STEP.search(path)[1]), path, Path(directory)))
    return [f for f in found if f[1] % every == 0]


def run_files(ckpt_path, run_dir):
    """Small files to upload once per run: (local path, path inside the run folder)."""
    directory = Path(ckpt_path).parent
    files = [(directory / name, name) for name in RUN_FILES if (directory / name).is_file()]
    if (directory / "reason1").is_dir():
        files += [(p, f"reason1/{p.name}") for p in sorted((directory / "reason1").iterdir()) if p.is_file()]
    if (run_dir / "command.txt").is_file():
        files.append((run_dir / "command.txt", "command.txt"))
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", action="append", default=[], help="dir whose <run>/checkpoints/*/ hold checkpoints")
    ap.add_argument("--extra", action="append", default=[], help="NAME=DIR: also upload the checkpoints in DIR as run NAME")
    ap.add_argument("--repo", default="structwam-ckpts", help="repo name, or user/name")
    ap.add_argument("--every", type=int, default=2000, help="upload only steps divisible by this")
    ap.add_argument("--poll", type=int, default=600, help="seconds between scans")
    ap.add_argument("--settle", type=int, default=600, help="seconds a file must be unmodified before upload")
    ap.add_argument("--max-staged", type=int, default=4, help="cap on hard-linked files waiting for upload")
    ap.add_argument("--state", required=True)
    ap.add_argument("--staging", required=True)
    ap.add_argument("--once", action="store_true", help="one scan, then exit")
    ap.add_argument("--dry-run", action="store_true", help="scan and report only: no links, no network")
    args = ap.parse_args()
    extras = [tuple(e.split("=", 1)) for e in args.extra]

    api = repo_id = None
    if not args.dry_run:
        from huggingface_hub import CommitOperationAdd, HfApi

        api = HfApi()
        user = api.whoami()["name"]
        repo_id = args.repo if "/" in args.repo else f"{user}/{args.repo}"
        api.create_repo(repo_id, private=True, exist_ok=True)
        if not api.repo_info(repo_id).private:
            sys.exit(f"{repo_id} is public; refusing to upload. Make it private or pass another --repo.")
        log(f"uploading to private repo {repo_id} as {user}")

    staging = Path(args.staging)
    state = load_state(args.state)
    while True:
        pending = []
        for run, step, path, run_dir in discover(args.root, extras, args.every):
            key = f"{run}/checkpoint_step_{step}.safetensors"
            if key in state["uploaded"]:
                continue
            try:
                st = os.stat(path)
            except FileNotFoundError:
                continue
            if time.time() - st.st_mtime >= args.settle:
                pending.append((st.st_mtime, key, run, step, path, run_dir, st.st_size))
        for _, key, run, step, path, run_dir, size in sorted(pending):
            gb = size / 1e9
            if args.dry_run:
                log(f"[dry-run] would upload {key} ({gb:.1f} GB) from {path}; "
                    f"run files: {[name for _, name in run_files(path, run_dir)]}")
                continue
            link = staging / key
            if not link.exists():
                if len(list(staging.glob("*/*.safetensors"))) >= args.max_staged:
                    log(f"staging full ({args.max_staged}); {key} waits")
                    continue
                link.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(path, link)
                except FileNotFoundError:
                    log(f"{key} was rotated away before it could be staged")
                    continue
            try:
                if run not in state["run_files"]:
                    ops = [CommitOperationAdd(path_in_repo=f"{run}/{name}", path_or_fileobj=str(local))
                           for local, name in run_files(path, run_dir)]
                    if ops:
                        api.create_commit(repo_id, operations=ops, commit_message=f"{run}: config and metadata")
                    state["run_files"].append(run)
                    save_state(args.state, state)
                t0 = time.time()
                api.upload_file(path_or_fileobj=str(link), path_in_repo=key, repo_id=repo_id,
                                commit_message=f"{run} step {step}")
            except Exception as e:  # network or server trouble: keep the link, retry on the next scan
                log(f"upload failed for {key}: {type(e).__name__}: {str(e)[:300]}")
                continue
            link.unlink()
            state["uploaded"][key] = {"bytes": size, "seconds": round(time.time() - t0),
                                      "at": time.strftime("%m-%d %H:%M")}
            save_state(args.state, state)
            log(f"uploaded {key} ({gb:.1f} GB) in {time.time() - t0:.0f}s")
        if args.once:
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
