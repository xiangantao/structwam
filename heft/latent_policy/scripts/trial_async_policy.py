"""Bounded offline trial: six frozen extraction GPUs feeding one ACT learner, no DDP."""

import argparse
import copy
import json
import queue
import threading
import time
import traceback
from pathlib import Path

import draccus
import torch
from lerobot.configs import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from torch.utils.data import default_collate

from lerobot_policy_latent_act import LatentACTConfig  # noqa: F401 -- register policy
from lerobot_policy_latent_act.encoders.wan_process import WanProcessEncoder
from lerobot_policy_latent_act.modeling_latent_act import LatentACTPolicy


class ReadyFeatures:
    """One-use visual runtime; all actual RGB encoding happened on the producer GPUs."""

    def __init__(self):
        self.features = None

    def encode(self, images):
        if self.features is None:
            raise RuntimeError("No queued features supplied")
        result, self.features = self.features, None
        if any(x.shape[0] != y.shape[0] for x, y in zip(result, images, strict=True)):
            raise ValueError("Features and labels have different batch sizes")
        return result

    def close(self):
        self.features = None


class ReadyBatchQueue:
    """Consume any ready member of this update; hold future updates without mixing labels."""

    def __init__(self, capacity):
        self.queue = queue.Queue(maxsize=capacity)
        self.pending = {}
        self.seen = set()

    def put(self, payload, timeout):
        self.queue.put(payload, timeout=timeout)

    def get(self, step, check_errors, timeout=1000):
        deadline = time.monotonic() + timeout
        while True:
            check_errors()
            for key in self.pending:
                if key[0] == step:
                    return self.pending.pop(key)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Feature queue timed out")
            try:
                payload = self.queue.get(timeout=min(1, remaining))
            except queue.Empty:
                continue
            key = payload[:2]
            if key[0] < step or key in self.seen:
                raise ValueError(f"Stale or duplicate feature batch: {key}")
            self.seen.add(key)
            self.pending[key] = payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extract-gpus", default="0,1,2,4,6,7")
    parser.add_argument("--learner-gpu", type=int, default=5)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    cards = [int(x) for x in args.extract_gpus.split(",")]
    if len(set(cards)) != len(cards) or args.learner_gpu in cards:
        parser.error("Extraction and learner devices must be distinct")
    if min(args.steps, args.batch_size) < 1 or not cards:
        parser.error("Positive steps, batch size and extraction devices required")
    args.output.mkdir(parents=True, exist_ok=False)
    raw = json.loads(args.config.read_text())
    config = draccus.decode(PreTrainedConfig, raw["policy"])
    config.device = f"cuda:{args.learner_gpu}"
    if config.encoder != "wan":
        parser.error("This trial currently profiles the existing Wan worker protocol")
    torch.set_num_threads(2)
    torch.manual_seed(raw["seed"])
    torch.cuda.set_device(args.learner_gpu)
    meta = LeRobotDatasetMetadata(raw["dataset"]["repo_id"], root=raw["dataset"]["root"])
    delta = {key: [i / meta.fps for i in range(-15, 1)] for key in config.camera_keys}
    delta["action"] = [i / meta.fps for i in range(config.chunk_size)]
    ds_kwargs = {
        "repo_id": raw["dataset"]["repo_id"],
        "root": raw["dataset"]["root"],
        "episodes": raw["dataset"]["episodes"],
        "delta_timestamps": delta,
        "video_backend": "pyav",
        "return_uint8": True,
    }
    # Each controller has its own reader/decoder state. All indices are planned before work.
    dataset = LeRobotDataset(**ds_kwargs)
    count = args.steps * len(cards) * args.batch_size
    if count > len(dataset):
        parser.error("Trial requests more than one pass over the selected data")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(raw["seed"]))[
        :count
    ]
    jobs = indices.reshape(args.steps, len(cards), args.batch_size).tolist()
    del dataset
    policy = LatentACTPolicy(config).to(config.device).train()
    runtime = ReadyFeatures()
    policy.model._visual_runtime = runtime
    policy.model._runtime_device = torch.device(config.device)
    pre, post = make_pre_post_processors(config, dataset_stats=meta.stats)
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=config.optimizer_lr,
        weight_decay=config.optimizer_weight_decay,
    )
    stopped = threading.Event()
    slots = [threading.BoundedSemaphore(1) for _ in cards]
    outputs = ReadyBatchQueue(capacity=len(cards))
    errors = queue.Queue()
    initialized = [threading.Event() for _ in cards]
    start = threading.Event()
    services = []
    records = []
    record_lock = threading.Lock()

    def record(item):
        item = {"time": time.time(), **item}
        with record_lock:
            records.append(item)
            print(json.dumps(item), flush=True)
            with (args.output / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(item) + "\n")

    def producer(lane, gpu):
        service = None
        try:
            reader = LeRobotDataset(**ds_kwargs)
            service = WanProcessEncoder(
                copy.deepcopy(config), f"cuda:{gpu}", config.wan_runtime_python
            )
            services.append(service)
            record({"event": "ready", "gpu": gpu, "worker_log": str(service.log_path)})
            initialized[lane].set()
            while not start.wait(0.2):
                if stopped.is_set():
                    return
            for step in range(args.steps):
                while not slots[lane].acquire(timeout=0.2):
                    if stopped.is_set():
                        return
                if stopped.is_set():
                    return
                begun = time.perf_counter()
                ids = jobs[step][lane]
                batch = default_collate([reader[index] for index in ids])
                images = [batch.pop(key).numpy() for key in config.camera_keys]
                decoded = time.perf_counter()
                service.connection.send(("encode_profiled", images))
                reply = service._receive(timeout=900)
                received = time.perf_counter()
                features = [
                    torch.from_numpy(array).to(getattr(torch, dtype))
                    for array, dtype in reply.pop("features")
                ]
                # Only state/action/task fields travel to the learner; camera padding is irrelevant here.
                batch = {
                    key: batch[key]
                    for key in ("observation.state", "action", "action_is_pad", "task")
                }
                item = {
                    "event": "extract",
                    "step": step,
                    "lane": lane,
                    "gpu": gpu,
                    "data_s": decoded - begun,
                    "roundtrip_s": received - decoded,
                    "host_cast_s": time.perf_counter() - received,
                    **reply,
                }
                record(item)
                payload = (step, lane, ids, batch, features)
                while not stopped.is_set():
                    try:
                        outputs.put(payload, timeout=0.2)
                        break
                    except queue.Full:
                        pass
                del payload, features, batch, images, reply
        except Exception:  # noqa: BLE001 -- relay worker failures to the learner
            errors.put(traceback.format_exc())
            stopped.set()
        finally:
            if service is not None:
                service.close()

    def check_errors():
        if not errors.empty():
            raise RuntimeError(errors.get())

    threads = [
        threading.Thread(target=producer, args=(i, gpu), daemon=True) for i, gpu in enumerate(cards)
    ]
    manifest = {
        "source_config": str(args.config.resolve()),
        "extract_gpus": cards,
        "learner_gpu": args.learner_gpu,
        "batch_size": args.batch_size,
        "accumulation": len(cards),
        "effective_batch_size": args.batch_size * len(cards),
        "steps": args.steps,
        "sample_indices": jobs,
        "extract_lanes": config.extract_lanes,
        "extract_batch_size": config.extract_batch_size,
        "consumption_order": "ready first within fixed global update",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + 1000
        while not all(event.is_set() for event in initialized):
            check_errors()
            if time.monotonic() > deadline:
                raise TimeoutError("Producers did not initialize")
            stopped.wait(0.2)
        start.set()
        wall_start = time.perf_counter()
        for step in range(args.steps):
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            total_loss = wait_s = h2d_s = compute_s = 0.0
            consumed_lanes = []
            for _ in cards:
                waited = time.perf_counter()
                got_step, lane, ids, batch, features = outputs.get(step, check_errors)
                wait_s += time.perf_counter() - waited
                if got_step != step or lane in consumed_lanes or ids != jobs[step][lane]:
                    raise RuntimeError("Features arrived with incorrect sample IDs")
                consumed_lanes.append(lane)
                record({"event": "consume", "step": step, "lane": lane, "gpu": cards[lane]})
                transfer = time.perf_counter()
                batch = pre(batch)
                runtime.features = [x.to(config.device) for x in features]
                # Shape-only handles; ReadyFeatures consumes actual producer maps, never these pixels.
                for key in config.camera_keys:
                    batch[key] = torch.zeros(1, device=config.device).expand(
                        args.batch_size, 16, 3, 256, 256
                    )
                del features
                torch.cuda.synchronize()
                h2d_s += time.perf_counter() - transfer
                slots[lane].release()
                compute = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = policy(batch)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite ACT loss")
                (loss / len(cards)).backward()
                total_loss += loss.item() / len(cards)
                torch.cuda.synchronize()
                compute_s += time.perf_counter() - compute
                del batch, loss
            compute = time.perf_counter()
            norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                config.get_optimizer_preset().grad_clip_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            torch.cuda.synchronize()
            compute_s += time.perf_counter() - compute
            record(
                {
                    "event": "update",
                    "step": step + 1,
                    "consumed_lanes": consumed_lanes,
                    "loss": total_loss,
                    "grad_norm": norm.item(),
                    "queue_wait_s": wait_s,
                    "h2d_s": h2d_s,
                    "act_compute_s": compute_s,
                    "step_s": time.perf_counter() - step_start,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
            )
        elapsed = time.perf_counter() - wall_start
        policy.save_pretrained(args.output / "checkpoint")
        pre.save_pretrained(args.output / "checkpoint")
        post.save_pretrained(args.output / "checkpoint")
        updates = [x for x in records if x["event"] == "update"]
        steady = updates[1:]
        steady_seconds = sum(x["step_s"] for x in steady)
        summary = {
            "total_samples": count,
            "pipeline_elapsed_s": elapsed,
            "end_to_end_samples_per_s": count / elapsed,
            "updates": updates,
            "steady_samples_per_s": len(steady) * args.batch_size * len(cards) / steady_seconds
            if steady
            else None,
            "checkpoint_scope": "policy and processors only; no optimizer resume state",
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        record({"event": "complete", "samples": count, "elapsed_s": elapsed})
    finally:
        stopped.set()
        start.set()
        # Abort only owned subprocesses on failure, avoiding a long join on an in-flight GPU call.
        for service in services:
            process = service.process
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
        for thread in threads:
            thread.join(timeout=30)
        policy.close()


if __name__ == "__main__":
    main()
