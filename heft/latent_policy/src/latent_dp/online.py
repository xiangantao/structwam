"""Online policy features: the frozen VACE student and the post-processing in the training loop.

``OnlineFeatures`` turns uint8 33-frame windows (one per camera and sample) into the policy
matrices of ``policy_features`` / ``finalize_policy_features``: K blocks [8, 42] (appearance
as a PCA of block-mean K, fitted once on warm-up windows), point tracks [32, 12, 3] and
their owners.  Nothing is cached between steps.
"""

import json
from multiprocessing import get_context

import numpy as np
import torch

from vace_latent_distill import fast_grid, fast_rope
from vace_latent_distill.config import ModelConfig
from vace_latent_distill.model import StructuredStudent
from vace_latent_distill.policy_features import BLOCKS, batch_features, block_inputs, blocks_job

KEYS = ("seg_k_l22h9", "seg_k_l15h7", "track_k", "text_attention", "depth")
APPEARANCE = 6
PROMPT_DIMS = 64


def pca(values, dims):
    mean = values.mean(0)
    _, _, vt = np.linalg.svd(values - mean, full_matrices=False)
    return mean.astype(np.float32), vt[:dims].astype(np.float32)


def prompt_table(words, text_cache, tasks):
    """[len(tasks), 3, 66]: per selected word, PCA-64 of its mean UMT5 embedding, position, valid."""
    names = sorted(words)
    vectors, places, owners = [], [], []
    for name in names:
        embedding = text_cache[name].float().numpy()
        length = max(int((np.abs(embedding).sum(-1) != 0).sum()), 1)
        for slot, word in enumerate(words[name]["words"]):
            vectors.append(embedding[word].mean(0))
            places.append(word[0] / length)
            owners.append((name, slot))
    mean, basis = pca(np.stack(vectors), PROMPT_DIMS)
    table = np.zeros((len(tasks), 3, PROMPT_DIMS + 2), np.float32)
    row = {task: i for i, task in enumerate(tasks)}
    for vector, place, (name, slot) in zip(vectors, places, owners):
        if name in row:
            table[row[name], slot, :PROMPT_DIMS] = (vector - mean) @ basis.T
            table[row[name], slot, PROMPT_DIMS] = place
            table[row[name], slot, PROMPT_DIMS + 1] = 1.0
    return table


class OnlineFeatures:
    def __init__(self, checkpoint, words_path, device, block_workers=8):
        saved = torch.load(str(checkpoint), map_location="cpu", weights_only=False, mmap=True)
        self.student = StructuredStudent(ModelConfig(**saved["config"]["model"]), pretrained=False)
        self.student.load_state_dict(saved["model"], strict=True)
        self.text_cache = torch.load(saved["config"]["train"]["text_cache"], map_location="cpu", weights_only=True)
        del saved
        self.student = self.student.eval().requires_grad_(False).to(device)
        # Inference speed-ups (same math; bf16 summation order differs slightly in attention).
        fast_rope.install(self.student.backbone, half_split=True)
        fast_grid.install(self.student)
        self.device = device
        self.words = json.loads(open(words_path).read())
        self.text = {}
        self.pool = get_context("spawn").Pool(block_workers)
        self.k_mean = self.k_basis = None

    def text_for(self, tasks):
        for task in set(tasks) - set(self.text):
            self.text[task] = self.text_cache[task].float().to(self.device)
        text = torch.stack([self.text[t] for t in tasks])
        return text, text.abs().sum(-1) != 0

    @torch.no_grad()
    def raw(self, rgb, tasks, wrist):
        """rgb [W, 33, 3, H, W] uint8 on device; tasks / wrist per window -> list of feature dicts."""
        text, mask = self.text_for(tasks)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.student(rgb, text, mask, observed_heads=("tracking",))
        futures = [{k: output["future"][k][i] for k in KEYS} for i in range(len(tasks))]
        jobs = [(*block_inputs(f, self.words[t]["words"]), bool(s)) for f, t, s in zip(futures, tasks, wrist)]
        return batch_features(output["observed"]["track_k"], output["future"]["track_k"], self.pool.map(blocks_job, jobs))

    def fit_appearance(self, results):
        k = np.concatenate([r["k"][r["valid"]].astype(np.float32) for r in results])
        self.k_mean, self.k_basis = pca(k, APPEARANCE)

    def assemble(self, results):
        """List of feature dicts -> objects [W, 8, 42], tracks [W, 32, 12, 3], owner [W, 32] (numpy)."""
        objects = np.zeros((len(results), BLOCKS, 42), np.float32)
        for i, r in enumerate(results):
            appearance = (r["k"].astype(np.float32) - self.k_mean) @ self.k_basis.T
            block = np.concatenate((r["valid"][:, None].astype(np.float32), r["centre"], r["size"], appearance,
                                    r["depth"], r["relevance"], r["trajectory"].reshape(BLOCKS, 24),
                                    r["confidence"][:, None]), -1)
            block[~r["valid"]] = 0
            objects[i] = block
        tracks = np.stack([r["tracks"].astype(np.float32) for r in results])
        owner = np.stack([r["owner"].astype(np.int64) for r in results])
        return objects, tracks, owner

    def close(self):
        self.pool.close()


# ----------------------------------------------------------------------------- policy maps (A + C + trunk)

def maps_job(job):
    """Process-pool entry: one window's maps (A) and chained keypoints (C), no diagnostics.

    Role mode returns ``(maps, keypoints)``; word mode also the selected words
    ``[(word, concentration)]`` and their token indices.
    """
    from .policy_maps import window_maps, word_window_maps

    torch.set_num_threads(1)
    fields, tokens, wrist, k_basis, mode = job
    if mode == "words":
        maps, keypoints, diagnostics = word_window_maps(fields, tokens, wrist, None, k_basis)
        return maps, keypoints, diagnostics["selected"], diagnostics["selected_tokens"]
    maps, keypoints, _ = window_maps(fields, tokens, wrist, None, k_basis)
    return maps, keypoints


class OnlineMaps:
    """Frozen student -> policy maps (A), keypoint tokens (C) and the V-JEPA2 trunk, per window.

    Dense K-K tracks, fixed-position K similarity and the HEFT keypoint tracks run batched on
    the device; segments and roles / words run in CPU workers.  Nothing is cached between steps.

    ``roles_path`` given: role mode (grammar roles, ``MAP_CHANNELS``; the existing DP runs).
    ``roles_path=None``: word mode (no grammar, ``WORD_MAP_CHANNELS``): instruction words come
    from the tokenizer, the most concentrated ones are picked per window, and instructions
    missing from the student's text cache are encoded with UMT5 on the fly.
    """

    def __init__(self, checkpoint, roles_path=None, device="cuda", workers=16, *, text_root=None):
        saved = torch.load(str(checkpoint), map_location="cpu", weights_only=False, mmap=True)
        self.student = StructuredStudent(ModelConfig(**saved["config"]["model"]), pretrained=False)
        self.student.load_state_dict(saved["model"], strict=True)
        self.text_cache = torch.load(saved["config"]["train"]["text_cache"], map_location="cpu", weights_only=True)
        self.text_root = text_root or saved["config"]["teacher"]["text_root"]
        del saved
        self.student = self.student.eval().requires_grad_(False).to(device)
        fast_rope.install(self.student.backbone, half_split=True)
        fast_grid.install(self.student)
        self.device = device
        self.mode = "roles" if roles_path else "words"
        self.roles = json.loads(open(roles_path).read()) if roles_path else None
        self.words = {}
        self._tokenizer = self._encoder = None
        self.text = {}
        self.pool = get_context("spawn").Pool(workers)
        self.k_basis = None

    @property
    def map_channels(self):
        from .policy_maps import MAP_CHANNELS, WORD_MAP_CHANNELS

        return WORD_MAP_CHANNELS if self.mode == "words" else MAP_CHANNELS

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from pathlib import Path

            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(Path(self.text_root) / "tokenizer", local_files_only=True)
        return self._tokenizer

    @torch.no_grad()
    def _encode(self, task):
        """UMT5 embedding of an instruction outside the cache, in the cache's format (padding rows zero)."""
        from pathlib import Path

        from transformers import UMT5EncoderModel

        if self._encoder is None:
            self._encoder = UMT5EncoderModel.from_pretrained(
                Path(self.text_root) / "text_encoder", torch_dtype=torch.bfloat16, local_files_only=True
            ).to(self.device).eval()
        tokens = self.tokenizer([task], padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        mask = tokens["attention_mask"].to(self.device)
        hidden = self._encoder(input_ids=tokens["input_ids"].to(self.device), attention_mask=mask).last_hidden_state
        return (hidden[0].float() * mask[0, :, None]).cpu()

    def text_for(self, tasks):
        for task in set(tasks) - set(self.text):
            embedding = self.text_cache.get(task)
            if embedding is None:
                if self.mode != "words":
                    raise KeyError(f"instruction not in the student's text cache: {task!r}")
                embedding = self._encode(task)
            self.text[task] = embedding.float().to(self.device)
        text = torch.stack([self.text[t] for t in tasks])
        return text, text.abs().sum(-1) != 0

    def words_for(self, task):
        """Word mode: ``content_words`` of an instruction (cached per instruction)."""
        from .policy_maps import content_words

        if task not in self.words:
            self.words[task] = content_words(task, self.tokenizer)[0]
        return self.words[task]

    @torch.no_grad()
    def __call__(self, rgb, tasks, wrist):
        """rgb [W, 33, 3, H, W] uint8 on device; tasks / wrist per window.

        Returns maps [W, len(self.map_channels), 32, 32], keypoint tokens [W, P, D] and their use
        mask [W, P] (numpy), the trunk [W, 17, 16, 16, 1024] (device, bf16) and present-K samples
        [W, 64, 256] for fitting the appearance PCA.  Word mode adds a sixth item: the selected
        words per window, ``{"embedding": [W, WORD_SLOTS, 4096] (device, float), "score": [W,
        WORD_SLOTS] concentration, "mask": [W, WORD_SLOTS] bool, "text": [[word, ...]]}`` in the
        order of the ``word_k`` map channels.
        """
        from vace_latent_distill.policy_features import track_many

        from .policy_maps import PRESENT, dense_tracks, fixed_position, keypoint_tokens, present_k, refine_keypoints

        text, mask = self.text_for(tasks)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.student(rgb, text, mask, observed_heads=("tracking", "semantic"))
        observed, future = output["observed"], output["future"]
        track_k = torch.cat((observed["track_k"][:, 4:12], future["track_k"][:, 8:13]), 1).float()
        dense = {k: v.cpu().numpy() for k, v in dense_tracks(track_k).items()}
        still, change = (v.cpu().numpy() for v in fixed_position(track_k))
        now = torch.cat((observed["text_attention"][:, 8:12], future["text_attention"][:, 8:9]), 1).float().mean(1)
        ahead = future["text_attention"][:, 12].float()
        seg = {key: future[key][:, PRESENT].float().cpu().numpy() for key in ("seg_k_l22h9", "seg_k_l15h7")}
        depth = future["depth"][:, PRESENT].float().cpu().numpy()
        jobs = []
        for w, task in enumerate(tasks):
            if self.mode == "words":
                tokens = self.words_for(task)
                length = 1 + max((t for _, ids in tokens for t in ids), default=0)
            else:
                tokens, length = self.roles[task]["tokens"], self.roles[task]["length"]
            fields = {"text_attention": torch.stack((now[w, :, :length], ahead[w, :, :length])).cpu().numpy(),
                      "seg_k_l22h9": seg["seg_k_l22h9"][w], "seg_k_l15h7": seg["seg_k_l15h7"][w], "depth": depth[w],
                      "dense": {k: v[w] for k, v in dense.items()}, "still": still[w], "change": change[w]}
            jobs.append((fields, tokens, bool(wrist[w]), self.k_basis, self.mode))
        results = self.pool.map(maps_job, jobs)
        maps = np.stack([r[0] for r in results])
        keypoints = [r[1] for r in results]
        # HEFT K-K keypoint tracks for all windows in one batched pass.
        count = len(keypoints[0]["role"])
        starts = np.zeros((len(tasks), count, 2), np.float32)
        used = [int((k["role"] >= 0).sum()) for k in keypoints]
        for w, k in enumerate(keypoints):
            cells = k["cell"][: used[w]]
            if used[w]:
                starts[w, : used[w]] = np.stack((cells % 32, cells // 32), -1) * 8 + 3.5
                starts[w, used[w]:] = starts[w, 0]
        forward, forward_sim = track_many(track_k, starts, tuple(range(PRESENT, 13)))
        backward, backward_sim = track_many(track_k, starts, tuple(range(PRESENT, -1, -1)))
        pixels = torch.cat((backward.flip(2)[:, :, :PRESENT], forward), 2).cpu().numpy()
        similarity = torch.cat((backward_sim.flip(2)[:, :, :PRESENT], forward_sim), 2).cpu().numpy()
        tokens, use = [], []
        for w, k in enumerate(keypoints):
            if used[w]:
                refine_keypoints(k, pixels[w, : used[w]], similarity[w, : used[w]], bool(wrist[w]))
            t, u = keypoint_tokens(k)
            tokens.append(t)
            use.append(u)
        k_sample = present_k(future["seg_k_l22h9"][:, PRESENT], future["seg_k_l15h7"][:, PRESENT])
        pick = torch.randperm(k_sample.shape[1])[:64]
        out = (maps, np.stack(tokens), np.stack(use), output["trunk"], k_sample[:, pick].numpy())
        if self.mode != "words":
            return out
        return out + (self._word_outputs(results, text),)

    def _word_outputs(self, results, text):
        """Selected words per window -> embeddings (mean of their UMT5 token rows), scores, mask."""
        from .policy_maps import WORD_SLOTS

        count = len(results)
        embedding = torch.zeros(count, WORD_SLOTS, text.shape[-1], device=text.device)
        score = np.zeros((count, WORD_SLOTS), np.float32)
        mask = np.zeros((count, WORD_SLOTS), bool)
        words = []
        for w, result in enumerate(results):
            selected, token_lists = result[2], result[3]
            words.append([word for word, _ in selected])
            for slot, ((_, concentration), ids) in enumerate(zip(selected, token_lists)):
                embedding[w, slot] = text[w, ids].mean(0)
                score[w, slot], mask[w, slot] = concentration, True
        return {"embedding": embedding, "score": score, "mask": mask, "text": words}

    def fit_appearance(self, k_samples):
        """Fixed PCA of present K; components scaled to unit variance."""
        from .policy_maps import APPEARANCE

        values = np.concatenate([k.reshape(-1, k.shape[-1]) for k in k_samples]).astype(np.float64)
        mean = values.mean(0)
        _, singular, vt = np.linalg.svd(values - mean, full_matrices=False)
        scale = singular[:APPEARANCE] / np.sqrt(len(values))
        self.k_basis = (mean.astype(np.float32), (vt[:APPEARANCE] / scale[:, None]).astype(np.float32))

    def close(self):
        self.pool.close()
