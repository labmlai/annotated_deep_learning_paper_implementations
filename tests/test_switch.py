"""Capacity overflow must bypass the expert branch through the outer residual."""

import copy

import pytest
import torch

from labml_nn.transformers.switch import SwitchFeedForward, SwitchTransformerLayer


class ScaleExpert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor([2.0, 3.0, 4.0]))

    def forward(self, x):
        return x * self.scale


class ZeroAttention(torch.nn.Module):
    def forward(self, *, query, key, value, mask=None):
        return torch.zeros_like(query)


def make_feedforward(capacity_factor, scale_prob, drop_tokens=True):
    feedforward = SwitchFeedForward(capacity_factor=capacity_factor, drop_tokens=drop_tokens,
                                   is_scale_prob=scale_prob, n_experts=2, expert=ScaleExpert(), d_model=3).double()
    with torch.no_grad():
        feedforward.switch.weight.zero_()
        feedforward.switch.bias.copy_(torch.tensor([2.0, 1.0]))
    return feedforward


@pytest.mark.parametrize('capacity_factor', [0.0, 0.5, 1.0])
@pytest.mark.parametrize('scale_prob', [False, True])
def test_switch_overflow_preserves_residual_outputs_and_gradients(monkeypatch, capacity_factor, scale_prob):
    monkeypatch.setattr(torch, 'randperm', lambda n: torch.arange(n))
    feedforward = make_feedforward(capacity_factor, scale_prob)
    layer = SwitchTransformerLayer(d_model=3, attn=ZeroAttention(), feed_forward=feedforward,
                                   dropout_prob=0).double()
    reference = copy.deepcopy(layer)
    x = torch.tensor([[[1.0, 2.0, 4.0]], [[2.0, -1.0, 3.0]],
                      [[4.0, 0.0, 2.0]], [[-1.0, 3.0, 1.0]]], requires_grad=True, dtype=torch.float64)
    ref_x = x.detach().clone().requires_grad_()
    capacity = int(capacity_factor * len(x) / 2)

    actual, counts, probability_sums, n_dropped, selected_probability = layer(x=x, mask=None)
    # The accepted prefix uses its expert; overflow contributes only the outer
    # residual. This oracle never adds the normalized input to dropped tokens.
    z = reference.norm_ff(ref_x)
    gate = reference.feed_forward.softmax(reference.feed_forward.switch(z))[:, :, 0:1]
    gate = gate if scale_prob else gate / gate.detach()
    accepted = reference.feed_forward.experts[0](z[:capacity]) * gate[:capacity]
    expected = torch.cat([ref_x[:capacity] + accepted, ref_x[capacity:]], dim=0)
    weights = torch.arange(x.numel(), dtype=x.dtype).reshape_as(x) / 10
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()

    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(feedforward.switch.weight.grad, reference.feed_forward.switch.weight.grad,
                               rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(feedforward.switch.bias.grad, reference.feed_forward.switch.bias.grad,
                               rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(feedforward.experts[0].scale.grad, reference.feed_forward.experts[0].scale.grad,
                               rtol=1e-10, atol=1e-10)
    assert n_dropped == len(x) - capacity
    torch.testing.assert_close(counts, torch.tensor([4.0, 0.0], dtype=x.dtype))
    torch.testing.assert_close(probability_sums.sum(), torch.tensor(4.0, dtype=x.dtype))
    assert selected_probability.shape == (4,)


@pytest.mark.parametrize('scale_prob', [False, True])
def test_switch_disable_dropping_and_sufficient_capacity_agree(scale_prob):
    x = torch.tensor([[[1.0, 2.0, 3.0]], [[2.0, 4.0, 6.0]]], dtype=torch.float64)
    disabled = make_feedforward(0, scale_prob, drop_tokens=False)
    sufficient = make_feedforward(2, scale_prob)
    sufficient.load_state_dict(disabled.state_dict())

    out_disabled = disabled(x)
    out_sufficient = sufficient(x)

    for index in (0, 1, 2, 4):
        torch.testing.assert_close(out_disabled[index], out_sufficient[index], rtol=0, atol=0)
    assert out_disabled[3] == out_sufficient[3] == 0


def test_switch_overflow_keeps_load_balancing_probability_gradients():
    x = torch.tensor([[[1.0, 2.0, 3.0]], [[-1.0, 3.0, 0.0]]], dtype=torch.float64)
    feedforward = make_feedforward(0, True)

    output, counts, probability_sums, n_dropped, _ = feedforward(x)
    probability_sums[0].backward()

    assert torch.count_nonzero(output) == 0
    assert n_dropped == 2
    assert counts.sum() == 2
    assert torch.count_nonzero(feedforward.switch.weight.grad) > 0


def test_switch_overflow_counts_tokens_across_multiple_experts(monkeypatch):
    monkeypatch.setattr(torch, 'randperm', lambda n: torch.arange(n))
    feedforward = make_feedforward(0.5, True)
    with torch.no_grad():
        feedforward.switch.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]))
        feedforward.switch.bias.zero_()
    x = torch.tensor([[[1.0, 2.0, 3.0]], [[2.0, 3.0, 1.0]], [[3.0, 1.0, 2.0]],
                      [[-1.0, 2.0, 3.0]], [[-2.0, 3.0, 1.0]], [[-3.0, 1.0, 2.0]]], dtype=torch.float64)

    output, counts, _, n_dropped, _ = feedforward(x)

    assert n_dropped == 4
    torch.testing.assert_close(counts, torch.tensor([3.0, 3.0], dtype=x.dtype))
    assert torch.count_nonzero(output[[1, 2, 4, 5]]) == 0
    assert torch.count_nonzero(output[[0, 3]]) == 6
