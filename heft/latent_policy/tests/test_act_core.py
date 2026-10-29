from copy import deepcopy

import torch


def test_original_act_forward_backward_parity(core, original_core, tiny_config):
    original = original_core["ACT"](tiny_config)
    modified = core["ACT"](tiny_config)
    modified.load_state_dict(original.state_dict(), strict=True)
    batch = {
        "observation.state": torch.randn(2, 8),
        "observation.environment_state": torch.randn(2, 10),
        "observation.images": [torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32)],
        "action": torch.randn(2, 3, 7),
        "action_is_pad": torch.zeros(2, 3, dtype=torch.bool),
    }
    torch.manual_seed(7)
    expected = original(batch)[0]
    torch.manual_seed(7)
    actual = modified(batch)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for (name, parameter), (_, other) in zip(
        original.named_parameters(), modified.named_parameters(), strict=True
    ):
        if parameter.grad is not None:
            torch.testing.assert_close(other.grad, parameter.grad, msg=name, rtol=1e-5, atol=1e-7)


def test_sdpa_attention_output_and_gradient(core, tiny_config):
    for name in ("ACTEncoderLayer", "ACTDecoderLayer"):
        eager = core[name](tiny_config)
        fast_config = deepcopy(tiny_config)
        fast_config.need_attention_weights = False
        fast = core[name](fast_config)
        fast.load_state_dict(eager.state_dict())
        x = torch.randn(7, 2, 16, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        if name == "ACTEncoderLayer":
            args = {
                "pos_embed": torch.randn(7, 1, 16),
                "key_padding_mask": torch.tensor([[False] * 6 + [True], [False] * 7]),
            }
            expected, actual = eager(x, **args), fast(y, **args)
        else:
            memory = torch.randn(11, 2, 16)
            args = {
                "decoder_pos_embed": torch.randn(7, 1, 16),
                "encoder_pos_embed": torch.randn(11, 1, 16),
            }
            expected, actual = eager(x, memory, **args), fast(y, memory, **args)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        expected.square().sum().backward()
        actual.square().sum().backward()
        torch.testing.assert_close(y.grad, x.grad, rtol=1e-3, atol=3e-6)
