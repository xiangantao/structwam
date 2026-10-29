"""Chunked LIBERO-10 evaluation of StructWAM step 24000 over many policy servers.

OpenWAM's Labtasker protocol without a Labtasker server: sealed reset-only manifests
(build_manifests.sh) fix every trial's pre-reset RNG state, so short trial chunks on any server
reproduce the continuous 50-trial run, and one shared queue keeps every server busy.

    python eval_chunks.py --name t0_chunks --slots 6,6,7,7 --base-port 8970 --tasks 0 --trials 10:30

``--slots`` lists one NPU per server (repeat an NPU for several servers on it; a server takes
~26 GB of the 64 GB).  Finished chunks (results.json present) are skipped on rerun.
"""

import argparse
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

R = Path("/home/ma-user/work/2337888765")
W = R / "workspace"
EV = W / "runs/libero10_structwam_s24000_v10"
REPO = W / "StructWAM"
BENCH = REPO / "benchmarks/libero"
SITE = R / "envs/vace-lerobot/lib/python3.11/site-packages"
CKPT = W / "runs/structwam_step24000_eval_ckpt"
SUITE = "libero_10"
TOTAL_TRIALS = 50


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def start_server(gpu: int, port: int, out: Path) -> subprocess.Popen:
    command = [str(EV / "server_python.sh"), str(REPO / "scripts/deploy.py"),
               "--ckpt-dir", str(CKPT), "--ckpt-name", "checkpoint_step_24000.safetensors",
               "--device", "cuda:0", "--host", "127.0.0.1", "--port", str(port),
               "--denoise-steps", "10", "--denoise-mode", "sync", "--inference-mode", "sync",
               "--inference-horizon", "10", "--compile-enabled", "false",
               "optimization.dit_cache.enabled=false"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}
    handle = (out / "logs" / f"server_npu{gpu}_port{port}.log").open("w")
    return subprocess.Popen(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)


def listening(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def chunk_dir(out: Path, task: int, start: int, count: int) -> Path:
    return out / f"task_{task:02d}" / f"trials_{start:03d}_{start + count - 1:03d}"


def run_chunk(out: Path, task: int, start: int, count: int, port: int, attempts: int) -> bool:
    manifest = EV / "manifests" / f"{SUITE}_task{task:02d}.json"
    manifest_hash = json.loads(manifest.read_text())["manifest_hash"]
    result_dir = chunk_dir(out, task, start, count)
    config_root = out / "libero_config" / f"port{port}"
    config_root.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, (REPO, SITE, BENCH))),
           "LIBERO_PATH": str(SITE), "LIBERO_CONFIG_ROOT": str(config_root),
           "LIBERO_CONFIG_PATH": str(config_root), "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
           "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    command = [str(EV / "libero_python.sh"), str(BENCH / "single_eval.py"),
               "--config", str(BENCH / "policy_config.yml"), "--suite", SUITE, "--task-id", str(task),
               "--host", "127.0.0.1", "--port", str(port), "--trial-start", str(start),
               "--num-trials", str(count), "--result-dir", str(result_dir),
               "--manifest-file", str(manifest), "--manifest-hash", manifest_hash,
               "--total-trials", str(TOTAL_TRIALS)]
    for attempt in range(1, attempts + 1):
        client_log = out / "logs" / f"task{task:02d}_trials{start:03d}_attempt{attempt}.log"
        with client_log.open("w") as handle:
            code = subprocess.call(command, cwd="/tmp", env=env, stdout=handle, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL)
        if code == 0 and (result_dir / "results.json").exists():
            return True
        log(f"task {task} trials {start}+{count} port {port}: attempt {attempt} exit {code}")
    return False


def summarize(out: Path) -> dict:
    tasks = {}
    for path in sorted(out.glob("task_*/trials_*/results.json")):
        result = json.loads(path.read_text())
        row = tasks.setdefault(int(result["task_id"]), {"successes": 0, "trials": 0})
        row["successes"] += int(result["successes"])
        row["trials"] += int(result["trial_stop"]) - int(result["trial_start"])
    total = {"successes": sum(r["successes"] for r in tasks.values()),
             "trials": sum(r["trials"] for r in tasks.values())}
    summary = {"suite": SUITE, "tasks": tasks, "overall": total}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--slots", required=True, help="NPU per server, e.g. 0,0,1,1")
    parser.add_argument("--base-port", type=int, default=8970)
    parser.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--trials", default="0:50", help="start:stop")
    parser.add_argument("--chunk", type=int, default=5)
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--server-timeout", type=int, default=1200)
    args = parser.parse_args()

    out = EV / f"chunks_{args.name}"
    (out / "logs").mkdir(parents=True, exist_ok=True)
    slots = [int(s) for s in args.slots.split(",")]
    ports = [args.base_port + i for i in range(len(slots))]
    busy = [p for p in ports if listening(p)]
    if busy:
        raise SystemExit(f"ports already in use: {busy}")
    first, stop = (int(v) for v in args.trials.split(":"))
    jobs = queue.Queue()
    for start in range(first, stop, args.chunk):
        for task in (int(t) for t in args.tasks.split(",")):
            count = min(args.chunk, stop - start)
            if not (chunk_dir(out, task, start, count) / "results.json").exists():
                jobs.put((task, start, count))
    log(f"{jobs.qsize()} chunk(s) over {len(slots)} server(s) on NPUs {args.slots}; output {out}")

    servers = [start_server(gpu, port, out) for gpu, port in zip(slots, ports)]

    def shutdown(*_):
        for server in servers:
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        sys.exit(1)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    failed = []

    def worker(port: int, server: subprocess.Popen) -> None:
        deadline = time.time() + args.server_timeout
        while not listening(port):
            if server.poll() is not None or time.time() > deadline:
                log(f"server on port {port} did not come up (exit {server.poll()})")
                return
            time.sleep(5)
        log(f"server on port {port} ready")
        while True:
            try:
                task, start, count = jobs.get_nowait()
            except queue.Empty:
                return
            began = time.time()
            ok = run_chunk(out, task, start, count, port, args.attempts)
            if not ok:
                failed.append((task, start, count))
            log(f"task {task} trials {start}-{start + count - 1} on port {port}: "
                f"{'done' if ok else 'FAILED'} in {(time.time() - began) / 60:.1f} min")

    threads = [threading.Thread(target=worker, args=(port, server)) for port, server in zip(ports, servers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for server in servers:
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    summary = summarize(out)
    rate = summary["overall"]["successes"] / max(summary["overall"]["trials"], 1)
    log(f"overall {summary['overall']['successes']}/{summary['overall']['trials']} ({rate:.1%}); "
        f"failed chunks: {failed or 'none'}")
    return 0 if not failed and jobs.empty() else 1


if __name__ == "__main__":
    sys.exit(main())
