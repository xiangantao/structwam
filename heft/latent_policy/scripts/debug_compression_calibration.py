import os, sys, time
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, cv2, av
sys.path.insert(0, "/home/ma-user/work/2337888765/workspace/heft/latent_policy/src")
sys.path.insert(0, "/home/ma-user/work/2337888765/workspace/heft/latent_policy/scripts")
from eval_libero_closed_loop import make_env, observe, NO_OP
from train_latent_dp_online import load_frames
from libero.libero import benchmark
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def av1(img, crf):
    """[H, W, 3] uint8 RGB -> AV1 (libsvtav1, yuv420p, one intra frame) -> decoded RGB."""
    enc = av.CodecContext.create("libsvtav1", "w")
    enc.width, enc.height, enc.pix_fmt = img.shape[1], img.shape[0], "yuv420p"
    enc.time_base = av.time_base if False else __import__("fractions").Fraction(1, 10)
    enc.options = {"crf": str(crf), "preset": "8"}
    frame = av.VideoFrame.from_ndarray(img, format="rgb24").reformat(format="yuv420p")
    frame.pts = 0
    packets = list(enc.encode(frame)) + list(enc.encode(None))
    dec = av.CodecContext.create("libdav1d", "r")
    out = [f for p in packets for f in dec.decode(p)] + list(dec.decode(None))
    return out[-1].to_ndarray(format="rgb24")

def jpeg(img, q):
    ok, buf = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1]

def sharp(img):
    g = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.abs(cv2.Laplacian(g, cv2.CV_32F)).mean())

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
env, language = make_env(suite, 0)
sims = []
for k in range(3):
    env.seed(k); env.reset(); raw = env.set_init_state(suite.get_task_init_states(0)[k])
    for _ in range(10):
        raw, _, _, _ = env.step(NO_OP)
    images, _ = observe(raw)
    sims.append([images[c].transpose(1, 2, 0) for c in range(2)])
from pathlib import Path; frames, names = load_frames(Path("/home/ma-user/work/2337888765/workspace/datasets/libero"))
ti = [i for i, n in names.items() if n == language][0]
ds = LeRobotDataset("local/libero", root="/home/ma-user/work/2337888765/workspace/datasets/libero", video_backend="pyav")
starts = np.flatnonzero((frames["task"] == ti) & (frames["frame"] == 0))[:6]
data = [[(ds[int(i)][k].numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8) for k in ("observation.images.image", "observation.images.image2")] for i in starts]
for c, cam in enumerate(("third", "wrist")):
    print(f"[{cam}] dataset frames (AV1 from LeRobot) sharpness: {np.mean([sharp(d[c]) for d in data]):.2f}")
    print(f"[{cam}] sim raw: {np.mean([sharp(s[c]) for s in sims]):.2f}")
    for crf in (20, 30, 40, 50):
        print(f"[{cam}] sim av1 crf{crf}: {np.mean([sharp(av1(s[c], crf)) for s in sims]):.2f}")
    for q in (50, 75, 90):
        print(f"[{cam}] sim jpeg q{q}: {np.mean([sharp(jpeg(s[c], q)) for s in sims]):.2f}")
t = time.time()
for _ in range(10):
    av1(sims[0][0], 30)
print("av1 round trip per frame: %.3f s" % ((time.time() - t) / 10))
out = np.concatenate([np.concatenate((sims[0][c], av1(sims[0][c], 30), data[0][c]), 1) for c in range(2)], 0)
cv2.imwrite("/home/ma-user/work/2337888765/workspace/runs/libero_eval_20261018/diagnosis/compression_calibration.png", out[:, :, ::-1])
