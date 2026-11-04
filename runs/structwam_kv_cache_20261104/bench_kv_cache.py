"""NPU timing and action agreement of StructWAM inference modes: video steps 10 / 1 / 0, with and
without the video K/V cache (``structure.inference.video_kv_cache``).

Built like the LIBERO-plus eval server (scripts/deploy.py arguments, DiT cache off, 10 denoising
steps, cuda:0 mapped to the NPU by transfer_to_npu), fed real training samples (canvas first frame,
33-frame structure histories, prompt, model-space proprio).  Per sample: one untimed warm-up
generate, then every mode ``--repeats`` times.  Reports per mode the median device-synced wall time
per generate and its student / rest split; a least-squares split of the rest into
prep + n_full * full forward + n_action * action-only forward; and how far each mode's actions are
from the uncached ones (relative L2 over the chunk and over the 10 executed steps), next to the
seed-to-seed and repeat-to-repeat spread of the same mode.

    python bench_kv_cache.py --ckpt-dir DIR --ckpt-name NAME --out OUT.json [--samples 8] [--repeats 3]
"""

import argparse
import json
import statistics
import time

import numpy as np
import torch

# name, video steps, K/V cache, seed
MODES = [("v10", 10, False, 42), ("v1", 1, False, 42), ("v1kv", 1, True, 42), ("v0", 0, False, 42),
         ("v0kv", 0, True, 42), ("v1_seed43", 1, False, 43)]
PAIRS = [("v1kv", "v1"), ("v0kv", "v0"), ("v1", "v10"), ("v1kv", "v10"), ("v1_seed43", "v1")]


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-8))


def sync():
    torch.cuda.synchronize()  # the NPU queue under transfer_to_npu


def set_mode(arch, steps: int, cache: bool):
    arch.inference_cfg.update(video_steps_far=steps, video_steps_near=steps, video_kv_cache=cache, log_timing=True)


def build(ckpt_dir: str, ckpt_name: str):
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

    from openwam.structure import phase

    gate_forward = phase.PhaseGate.forward

    def gate(self, pooled, proprio=None):  # deploy-only fix, as in runs/libero10_structwam_s24000_v10/npu_eval_entry.py
        if proprio is not None:
            proprio = proprio.reshape(pooled.shape[0], -1)
        return gate_forward(self, pooled, proprio)

    phase.PhaseGate.forward = gate
    from openwam.deploy import server as deploy_server

    args = deploy_server._build_argparser().parse_args([
        "--ckpt-dir", ckpt_dir, "--ckpt-name", ckpt_name, "--device", "cuda:0", "--denoise-steps", "10",
        "--denoise-mode", "sync", "--inference-horizon", "10", "--compile-enabled", "false",
        "optimization.dit_cache.enabled=false",
    ])
    cfg, resolved = deploy_server.resolve_deploy_args(args)
    server = deploy_server.build_server_from_config(cfg=cfg, ckpt_dir=resolved, device="cuda:0", ckpt_name=ckpt_name)
    engine = server.engine
    # The samples already carry model-space proprio; skip the raw-state normalizer.
    engine.architecture.normalize_deploy_proprio = lambda p: torch.as_tensor(p, dtype=torch.float32).reshape(-1)
    return engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--ckpt-name", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    a = parser.parse_args()

    t0 = time.time()
    engine = build(a.ckpt_dir, a.ckpt_name)
    arch = engine.architecture
    print(f"loaded in {time.time() - t0:.0f}s on {arch.device}, dtype {arch.dtype}", flush=True)

    from omegaconf import OmegaConf

    from openwam.dataloader.registry import build_dataset

    data_cfg = OmegaConf.to_container(OmegaConf.load(f"{a.ckpt_dir}/config.yaml").dataloader, resolve=True)
    data_cfg["structure_augment"] = {"enabled": False}
    dataset = build_dataset(OmegaConf.create(data_cfg), split="train")
    indices = np.linspace(1000, len(dataset) - 1000, a.samples).astype(int).tolist()

    rows = []  # one per timed generate
    agreement = {f"{x}_vs_{y}": [] for x, y in PAIRS}
    agreement.update({f"{x}_vs_{y}_first10": [] for x, y in PAIRS})
    agreement["repeat_v1"] = []
    agreement["repeat_v1kv"] = []
    for index in indices:
        sample = dataset[index]
        conditions = {"first_frame_image": sample["first_frame_image"], "prompt": sample["prompt"],
                      "proprio": sample["proprio"], "structure_frames": sample["structure_frames"]}
        set_mode(arch, 10, False)
        engine.generate(dict(conditions, seed=0))  # warm-up: prompt embedding, kernels
        first, last = {}, {}
        for name, steps, cache, seed in MODES:
            set_mode(arch, steps, cache)
            for repeat in range(a.repeats):
                sync()
                start = time.perf_counter()
                result = engine.generate(dict(conditions, seed=seed))
                sync()
                timing = result.get("timing", {})
                rows.append({"sample": index, "mode": name, "repeat": repeat, "wall_s": time.perf_counter() - start,
                             "student_s": timing.get("student_s"), "rest_s": timing.get("denoise_s"),
                             "full": timing.get("full_forwards"), "action_only": timing.get("action_only_forwards")})
                actions = np.asarray(result["actions"], dtype=np.float64)
                first.setdefault(name, actions)
                last[name] = actions
        for x, y in PAIRS:
            agreement[f"{x}_vs_{y}"].append(rel(first[x], first[y]))
            agreement[f"{x}_vs_{y}_first10"].append(rel(first[x][:10], first[y][:10]))
        agreement["repeat_v1"].append(rel(last["v1"], first["v1"]))
        agreement["repeat_v1kv"].append(rel(last["v1kv"], first["v1kv"]))
        done = [r for r in rows if r["sample"] == index]
        print(f"sample {index}: " + ", ".join(
            f"{name} {statistics.median(r['wall_s'] for r in done if r['mode'] == name):.3f}s" for name, *_ in MODES),
            flush=True)

    summary = {}
    for name, steps, cache, _ in MODES:
        mine = [r for r in rows if r["mode"] == name]
        summary[name] = {"video_steps": steps, "kv_cache": cache, "n": len(mine),
                         "wall_s": statistics.median(r["wall_s"] for r in mine),
                         "student_s": statistics.median(r["student_s"] for r in mine),
                         "rest_s": statistics.median(r["rest_s"] for r in mine),
                         "full_forwards": mine[0]["full"], "action_only_forwards": mine[0]["action_only"]}
    # rest = prep + full * t_full + action_only * t_action, fitted over every timed generate
    design = np.array([[1.0, r["full"], r["action_only"]] for r in rows])
    target = np.array([r["rest_s"] for r in rows])
    (prep, t_full, t_action), *_ = np.linalg.lstsq(design, target, rcond=None)
    fit = {"prep_s": float(prep), "full_forward_s": float(t_full), "action_only_forward_s": float(t_action)}
    speed = {f"{name}_speedup_vs_v10": summary["v10"]["wall_s"] / summary[name]["wall_s"]
             for name in ("v1", "v1kv", "v0", "v0kv")}
    agree = {key: {"mean": float(np.mean(v)), "max": float(np.max(v))} for key, v in agreement.items()}
    report = {"ckpt": f"{a.ckpt_dir}/{a.ckpt_name}", "samples": indices, "repeats": a.repeats,
              "summary": summary, "forward_fit": fit, "speedup": speed, "agreement": agree, "rows": rows}
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)

    print("\nmode       video  cache  full+action  wall(s)  student(s)  rest(s)")
    for name, s in summary.items():
        print(f"{name:<10} {s['video_steps']:>5}  {str(s['kv_cache']):<5}  {s['full_forwards']:>4}+{s['action_only_forwards']:<6}"
              f"  {s['wall_s']:7.3f}  {s['student_s']:10.3f}  {s['rest_s']:7.3f}")
    print(f"\nrest = {prep:.3f}s prep + n_full x {t_full:.3f}s + n_action x {t_action:.3f}s")
    print("speedup vs v10: " + ", ".join(f"{k.split('_')[0]} {v:.2f}x" for k, v in speed.items()))
    print("\naction agreement (relative L2, mean / max over samples):")
    for key, v in agree.items():
        print(f"  {key:<24} {v['mean']:.4f} / {v['max']:.4f}")
    print(f"\nwritten {a.out}", flush=True)


if __name__ == "__main__":
    main()
