#!/usr/bin/env python3
"""Extract loss-only VACE teachers, upload to TOS, and retain a bounded cache."""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent))

from vace_latent_distill.offline_store import ArtifactCache, TosStore, file_sha256


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def log(**values):
    print(json.dumps({"time": time.time(), **values}, allow_nan=False), flush=True)


def load_job(run):
    return json.loads((run / "job.json").read_text())


def parse_gpu_list(value):
    values = [int(item) for item in str(value).split(",") if item.strip()]
    if not values or len(values) != len(set(values)) or any(item < 0 for item in values):
        raise ValueError(f"Invalid GPU list: {value!r}")
    return values


def store_for(job):
    return TosStore(job["tos_config"], endpoint=job["tos_endpoint"], scratch=job["tos_scratch"])


def cache_for(job):
    return ArtifactCache(job["cache_dir"], job["cache_limit_bytes"])


def prepare(args):
    from vace_latent_distill.config import PipelineConfig
    from vace_latent_distill.offline import build_windows, provenance

    config = PipelineConfig.load(args.config)
    if config.model.task != "shared":
        raise ValueError("Offline extraction collects all five heads with task=shared")
    config.train.windows_per_episode_camera = 8
    identity = provenance(config, config.train.manifest, config.train.text_cache)
    rows = [json.loads(line) for line in Path(config.train.manifest).read_text().splitlines() if line.strip()]
    windows = build_windows(rows, seed=config.train.seed, count=8)
    for row in windows:
        row["provenance_id"] = identity["id"]
    job = {
        "schema": identity["schema"], "run_id": identity["id"],
        "prefix": args.tos_prefix.rstrip("/"), "config": config.to_dict(),
        "provenance": identity, "cache_dir": str(args.cache_dir.resolve()),
        "cache_limit_bytes": args.cache_limit_bytes,
        "tos_config": str(args.tos_config.resolve()), "tos_endpoint": args.tos_endpoint,
        "tos_scratch": str(args.run / "tos_transfers"),
        "windows": len(windows), "episode_camera_rows": len(rows),
        "gpus": parse_gpu_list(args.gpus), "uploaders": args.uploaders,
    }
    if not job["prefix"].startswith("tos://"):
        raise ValueError("Destination must be a TOS URI")
    args.run.mkdir(parents=True, exist_ok=True)
    old = args.run / "job.json"
    if old.exists() and json.loads(old.read_text()) != job:
        raise ValueError("Existing run has a different immutable extraction recipe")
    write_json(old, job)
    write_json(args.run / "pipeline.json", config.to_dict())
    write_json(args.run / "provenance.json", identity)
    temporary = args.run / "windows.jsonl.tmp"
    with temporary.open("w") as output:
        for row in windows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(args.run / "windows.jsonl")
    cache_for(job).register(job["run_id"], windows, job["prefix"])
    log(event="prepared", run=str(args.run), prefix=job["prefix"], windows=len(windows),
        cache_dir=job["cache_dir"], cache_limit_bytes=job["cache_limit_bytes"])


def extract_worker(args):
    import torch
    from vace_latent_distill.config import NEGATIVE_PROMPT, PipelineConfig
    from vace_latent_distill.data import decode_indices
    from vace_latent_distill.offline import pack_target, provenance
    from vace_latent_distill.teacher import VaceTeacher, content_token_masks

    job = load_job(args.run)
    config = PipelineConfig.load(args.run / "pipeline.json")
    if provenance(config, config.train.manifest, config.train.text_cache)["id"] != job["run_id"]:
        raise ValueError("Teacher sources/weights/text changed after this cache run was prepared")
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    free, _ = torch.cuda.mem_get_info()
    if free < 18 * 1024**3:
        raise RuntimeError("Less than18 GiB GPU memory free; extraction worker not started")
    cache = cache_for(job)
    text_cache = torch.load(config.train.text_cache, map_location="cpu", weights_only=True)
    prompts = {row["task"] for row in map(json.loads, (args.run / "windows.jsonl").read_text().splitlines())}
    content = content_token_masks(config.teacher.text_root, prompts)
    log(event="loading_teacher", gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), owner=args.owner)
    teacher = VaceTeacher(config.teacher, "shared", "cuda:0", text_cache[NEGATIVE_PROMPT])
    log(event="teacher_ready", owner=args.owner)
    try:
        while not (args.run / "STOP").exists():
            entry = cache.claim_extract(job["run_id"], args.owner)
            if entry is None:
                if cache.summary(job["run_id"])["states"].get("planned", 0) == 0:
                    break
                time.sleep(1)
                continue
            row = json.loads(entry["metadata"])
            started = time.monotonic()
            try:
                log(event="extract_start", sample_id=row["sample_id"], owner=args.owner)
                rgb = torch.from_numpy(decode_indices(row["video"], row["frame_indices"], row["fps"]))
                text = text_cache[row["task"]].float()
                mask = text.abs().sum(-1) != 0
                target = teacher.extract(rgb, text, mask, row["teacher_seed"], content[row["task"]])
                payload = pack_target(target, row, teacher.protocol, job["run_id"])
                del target, rgb
                path = Path(entry["path"])
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = Path(str(path) + ".partial")
                torch.save(payload, temporary)
                del payload
                sha = file_sha256(temporary)
                temporary.replace(path)
                cache.ready(entry["uri"], path, sha, args.owner)
                log(event="extracted", sample_id=row["sample_id"], seconds=time.monotonic() - started,
                    bytes=path.stat().st_size, owner=args.owner,
                    cuda_peak_bytes=torch.cuda.max_memory_allocated())
            except Exception as error:
                cache.failed(entry["uri"], args.owner, error)
                log(event="extract_error", sample_id=row["sample_id"], error=str(error), owner=args.owner)
                traceback.print_exc()
                if isinstance(error, torch.cuda.OutOfMemoryError):
                    raise
    finally:
        teacher.close()
    log(event="extract_worker_finished", owner=args.owner)


def upload_worker(args):
    job = load_job(args.run)
    cache, store = cache_for(job), store_for(job)
    while not (args.run / "STOP").exists():
        entry = cache.claim_upload(job["run_id"], args.owner)
        if entry is None:
            states = cache.summary(job["run_id"])["states"]
            if not any(states.get(key, 0) for key in ("planned", "extracting", "ready", "uploading")):
                break
            time.sleep(1)
            continue
        started = time.monotonic()
        try:
            sha = store.put(entry["path"], entry["uri"])
            cache.uploaded(entry["uri"], args.owner, sha)
            log(event="uploaded", uri=entry["uri"], bytes=entry["size"],
                seconds=time.monotonic() - started, sha256=sha, owner=args.owner)
        except Exception as error:
            cache.failed(entry["uri"], args.owner, error, upload=True)
            log(event="upload_error", uri=entry["uri"], error=str(error), owner=args.owner)
    log(event="upload_worker_finished", owner=args.owner)


def export_manifest(run, job, cache):
    rows = cache.export(job["run_id"])
    temporary = run / "uploaded.jsonl.tmp"
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(run / "uploaded.jsonl")
    return len(rows)


def coordinator(args):
    job = load_job(args.run)
    cache, store = cache_for(job), store_for(job)
    lock = (args.run / "coordinator.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (args.run / "STOP").unlink(missing_ok=True)
    previous = args.run / "processes.json"
    if previous.exists():
        for entry in json.loads(previous.read_text()):
            cache.recover_owner(entry["owner"])
    actors = []

    def spawn(role, name, gpu=None, retries=0):
        owner = f"{socket_name()}:{name}:{uuid.uuid4().hex}"
        environment = dict(os.environ)
        environment.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
        if gpu is not None:
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        output = (args.run / "logs" / f"{name}.log").open("a")
        process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
            "--role", role, "--run", str(args.run), "--owner", owner],
            env=environment, stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        output.close()
        entry = {"role": role, "name": name, "gpu": gpu, "owner": owner,
                 "pid": process.pid, "retries": retries, "process": process}
        actors.append(entry)
        write_json(args.run / "processes.json", [{key: value for key, value in actor.items() if key != "process"}
                                                 for actor in actors if actor["process"].poll() is None])
        return entry

    (args.run / "logs").mkdir(exist_ok=True)
    for gpu in job["gpus"]:
        spawn("worker", f"gpu{gpu}", gpu)
    for index in range(job["uploaders"]):
        spawn("uploader", f"uploader{index}")
    log(event="workers_started", gpus=job["gpus"], uploaders=job["uploaders"])
    published = -1
    reported_dead = set()
    try:
        for name in ("pipeline.json", "provenance.json", "windows.jsonl", "job.json"):
            store.put(args.run / name, job["prefix"] + "/meta/" + name)
        store.put(job["config"]["train"]["text_cache"], job["prefix"] + "/meta/text.pt")
        while True:
            stopped = (args.run / "STOP").exists()
            summary = cache.summary(job["run_id"])
            summary.update(time=time.time(), prefix=job["prefix"],
                           active=[{"name": actor["name"], "pid": actor["pid"]}
                                   for actor in actors if actor["process"].poll() is None])
            write_json(args.run / "progress.json", summary)
            exported = export_manifest(args.run, job, cache)
            log(event="progress", **summary)
            if exported >= published + 100 or exported == job["windows"]:
                store.put(args.run / "uploaded.jsonl", job["prefix"] + "/meta/uploaded.jsonl")
                published = exported
            if stopped:
                break
            for actor in list(actors):
                code = actor["process"].poll()
                if code is None or actor["owner"] in reported_dead:
                    continue
                reported_dead.add(actor["owner"])
                cache.recover_owner(actor["owner"])
                if code != 0:
                    log(event="worker_exited", name=actor["name"], code=code)
                    if actor["retries"] < 2:
                        spawn(actor["role"], actor["name"], actor["gpu"], actor["retries"] + 1)
            if not any(actor["process"].poll() is None for actor in actors):
                break
            time.sleep(10)
    finally:
        for actor in actors:
            if actor["process"].poll() is None:
                actor["process"].terminate()
        for actor in actors:
            try:
                actor["process"].wait(timeout=30)
            except subprocess.TimeoutExpired:
                actor["process"].kill()
                actor["process"].wait()
            cache.recover_owner(actor["owner"])
        export_manifest(args.run, job, cache)
        summary = cache.summary(job["run_id"])
        summary.update(time=time.time(), finished=summary["uploaded"] == job["windows"], active=[])
        write_json(args.run / "progress.json", summary)
        try:
            store.put(args.run / "uploaded.jsonl", job["prefix"] + "/meta/uploaded.jsonl")
            store.put(args.run / "progress.json", job["prefix"] + "/meta/progress.json")
        except Exception as error:
            log(event="final_manifest_upload_error", error=str(error))
        log(event="coordinator_finished", **summary)


def socket_name():
    import socket
    return socket.gethostname()


def start(args):
    job = load_job(args.run)
    if args.override_gpus is not None:
        # GPU assignment is operational metadata; the teacher/provenance recipe
        # remains immutable while an existing queue is resumed elsewhere.
        job["gpus"] = parse_gpu_list(args.override_gpus)
        write_json(args.run / "job.json", job)
    pid_path = args.run / "coordinator.pid"
    if pid_path.exists():
        pid = int(pid_path.read_text())
        cmdline = Path(f"/proc/{pid}/cmdline")
        if cmdline.exists() and str(args.run).encode() in cmdline.read_bytes():
            raise RuntimeError(f"This run is already managed by PID{pid}")
    (args.run / "logs").mkdir(exist_ok=True)
    with (args.run / "logs/coordinator.log").open("a") as output:
        process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
            "--role", "coordinator", "--run", str(args.run)],
            stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    pid_path.write_text(str(process.pid) + "\n")
    log(event="started", coordinator_pid=process.pid, run=str(args.run))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("prepare", "start", "coordinator", "worker", "uploader", "status", "export", "stop"), required=True)
    parser.add_argument("--run", type=Path, default=WORKSPACE / "runs/vace_teacher_offline_libero8_20261009")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/vace_structured.json")
    parser.add_argument("--cache-dir", type=Path, default=WORKSPACE / "cache/vace_teacher_libero8")
    parser.add_argument("--cache-limit-bytes", type=int, default=50_000_000_000)
    parser.add_argument("--tos-prefix", default="tos://kairos-vla-datas/research/2337888765/vace_teacher/libero_8offset_lossfp16_20261009_v1")
    parser.add_argument("--tos-config", type=Path, default=Path("/root/.tosutilconfig"))
    parser.add_argument("--tos-endpoint", default="tos-cn-shanghai.ivolces.com")
    parser.add_argument("--gpus", default="0,2,4,6")
    parser.add_argument("--override-gpus", default=None,
                        help="replace only the worker GPU list when resuming an existing run")
    parser.add_argument("--uploaders", type=int, default=2)
    parser.add_argument("--owner")
    args = parser.parse_args()
    args.run = args.run.resolve()
    if args.role == "prepare":
        prepare(args)
    elif args.role == "start":
        start(args)
    elif args.role == "coordinator":
        coordinator(args)
    elif args.role == "worker":
        extract_worker(args)
    elif args.role == "uploader":
        upload_worker(args)
    elif args.role == "stop":
        (args.run / "STOP").touch()
        log(event="stop_requested", run=str(args.run))
    else:
        job = load_job(args.run)
        cache = cache_for(job)
        if args.role == "export":
            log(event="manifest_exported", windows=export_manifest(args.run, job, cache))
        else:
            log(event="status", **cache.summary(job["run_id"]))


if __name__ == "__main__":
    main()
