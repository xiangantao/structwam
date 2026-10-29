"""Camera/chunk layout and autograd boundary of the frozen student adapter."""

from types import SimpleNamespace

import torch
from conftest import PLUGIN, load_file


def test_distilled_camera_chunk_order_and_trainable_projection(encoder_modules):
    module = load_file("latent_test_encoders.distilled", PLUGIN / "encoders/distilled.py")
    runtime = module.DistilledEncoder.__new__(module.DistilledEncoder)
    runtime.config = SimpleNamespace()
    runtime.device = torch.device("cpu")
    runtime.extract_batch_size = 3  # deliberately crosses the camera boundary

    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
            self.seen = []

        def feature_map(self, rgb):
            assert not self.training and not torch.is_grad_enabled()
            markers = rgb[:, 0, 0, 0, 0].float()
            self.seen.extend(markers.tolist())
            time = torch.arange(8).view(1, 1, 8, 1, 1)
            channel = torch.arange(1152).view(1, 1152, 1, 1, 1)
            return (markers[:, None, None, None, None] * 10000 + time * 1152 + channel).expand(-1, -1, -1, 16, 16)

    runtime.model = Student()
    images = [torch.stack([torch.full((16, 3, 256, 256), v, dtype=torch.uint8) for v in values]) for values in ((1, 2), (3, 4))]
    # Some policy evaluators use inference_mode. Returned maps must still work in a later training step.
    with torch.inference_mode():
        maps = runtime.encode(images)
    assert runtime.model.seen == [1, 2, 3, 4]
    assert [tuple(x.shape) for x in maps] == [(2, 8, 1152, 16, 16)] * 2
    assert all(not x.is_inference() and not x.requires_grad for x in maps)
    assert maps[0][1, 5, 19, 7, 9] == 20000 + 5 * 1152 + 19
    assert maps[1][0, 7, 31, 2, 8] == 30000 + 7 * 1152 + 31
    projection = torch.nn.Conv2d(1152, 4, 1)
    projection(maps[1][:1, :1].reshape(1, 1152, 16, 16)).square().mean().backward()
    assert projection.weight.grad is not None
    assert runtime.model.scale.grad is None
    assert runtime.health() == {"training": False, "any_requires_grad": False, "output_shape": [8, 1152, 16, 16]}
    runtime.close()
