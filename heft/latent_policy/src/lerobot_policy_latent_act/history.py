"""Episode-scoped camera history, using LeRobot's existing queue and sampling helpers."""

from collections import deque


class CameraHistory:
    def __init__(self, keys, frames: int):
        self.keys = tuple(keys)
        self.frames = frames
        self.reset()

    def reset(self):
        self.queues = {key: deque(maxlen=self.frames) for key in self.keys}
        self.batch_size = None
        self.tasks = None
        self.steps_seen = 0

    def update(self, batch):
        from lerobot.policies.utils import populate_queues

        snapshots = {}
        for key in self.keys:
            image = batch[key]
            if image.ndim != 4:
                raise ValueError(
                    "select_action requires current images [B,C,H,W], not a history window"
                )
            if self.batch_size is not None and image.shape[0] != self.batch_size:
                raise ValueError("Call policy.reset() before changing rollout batch size")
            self.batch_size = image.shape[0]
            snapshots[key] = image.detach().clone()
        tasks = batch.get("task")
        if isinstance(tasks, str):
            tasks = [tasks]
        tasks = tuple(tasks) if tasks is not None else None
        if self.steps_seen and tasks != self.tasks:
            raise ValueError(
                "Task changed without policy.reset(); queued actions/history would cross episodes"
            )
        self.tasks = tasks
        populate_queues(self.queues, snapshots)
        self.steps_seen += 1

    def stack(self, batch):
        from lerobot.policies.pi05.memory import sample_observation_history

        if not self.steps_seen:
            raise RuntimeError("No current observation in history")
        result = dict(batch)
        for key, queue in self.queues.items():
            result[key], result[f"{key}_is_pad"] = sample_observation_history(
                list(queue), num_frames=self.frames, stride=1, steps_seen=self.steps_seen
            )
        return result
