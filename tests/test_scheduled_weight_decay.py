"""Decoupled shrinkage must use the same scheduled rate as the update."""

import math

import pytest
import torch

from labml_nn.optimizers import WeightDecay
from labml_nn.optimizers.adam import Adam
from labml_nn.optimizers.adam_fp16 import AdamFP16
from labml_nn.optimizers.adam_warmup import AdamWarmup
from labml_nn.optimizers.adam_warmup_cosine_decay import AdamWarmupCosineDecay
from labml_nn.optimizers.ada_belief import AdaBelief
from labml_nn.optimizers.noam import Noam
from labml_nn.optimizers.radam import RAdam


def scheduled_rate(kind, step, maximum):
    if kind == "noam":
        return maximum / math.sqrt(16.) * min(step ** -.5, step / 4. ** 1.5)
    if kind in ("warmup", "cosine") and step < 4:
        return 1e-8 + step * maximum / 4.
    if kind == "cosine":
        return maximum * max(.1, .5 * (1. + math.cos(math.pi * (step - 4) / 8.)))
    return maximum


@pytest.mark.parametrize("kind", ["constant", "warmup", "noam", "cosine"])
@pytest.mark.parametrize("decay_kind", ["decoupled", "absolute", "coupled"])
@pytest.mark.parametrize("optimized", [False, True])
def test_scheduled_adam_trajectory_matches_independent_recurrence(kind, decay_kind, optimized):
    initial = torch.tensor([1.2, -.8], dtype=torch.float64)
    parameter = torch.nn.Parameter(initial.clone())
    maximum, coefficient, eps = .04, .2, .03
    beta1, beta2 = .6, .8
    decay = WeightDecay(weight_decay=coefficient, weight_decouple=decay_kind != "coupled",
                        absolute=decay_kind == "absolute")
    classes = {"constant": Adam, "warmup": AdamWarmup, "noam": Noam, "cosine": AdamWarmupCosineDecay}
    kwargs = {} if kind == "constant" else {"warmup": 4}
    if kind == "noam":
        kwargs['d_model'] = 16
    if kind == "cosine":
        kwargs['total_steps'] = 12
    optimizer = classes[kind]([parameter], lr=maximum, betas=(beta1, beta2), eps=eps,
                              optimized_update=optimized, weight_decay=decay, **kwargs)
    expected, momentum, variance = initial.clone(), torch.zeros_like(initial), torch.zeros_like(initial)
    gradients = [torch.tensor(value, dtype=torch.float64) for value in
                 [(0., 0.), (.2, -.3), (-.1, .4), (.05, -.02)]]
    for step in range(1, 13):
        gradient = gradients[(step - 1) % len(gradients)]
        parameter.grad = gradient.clone()
        optimizer.step()
        rate = scheduled_rate(kind, step, maximum)
        if decay_kind == "decoupled":
            expected = expected * (1. - rate * coefficient)
        elif decay_kind == "absolute":
            expected = expected * (1. - coefficient)
        else:
            gradient = gradient + coefficient * expected
        momentum = beta1 * momentum + (1. - beta1) * gradient
        variance = beta2 * variance + (1. - beta2) * gradient.square()
        if optimized:
            expected = expected - rate * math.sqrt(1. - beta2 ** step) / (1. - beta1 ** step) * momentum / (variance.sqrt() + eps)
        else:
            expected = expected - rate * momentum / (1. - beta1 ** step) / ((variance / (1. - beta2 ** step)).sqrt() + eps)
        torch.testing.assert_close(parameter, expected, rtol=2e-12, atol=2e-12)
        torch.testing.assert_close(optimizer.state[parameter]['exp_avg'], momentum, rtol=2e-12, atol=2e-12)
        torch.testing.assert_close(optimizer.state[parameter]['exp_avg_sq'], variance, rtol=2e-12, atol=2e-12)
        # The effective rate is local to the current update, not written over lr_max.
        assert optimizer.param_groups[0]['lr'] == maximum


@pytest.mark.parametrize("family", [Adam, RAdam, AdaBelief, AdamFP16])
@pytest.mark.parametrize("skip_gradients", [False, True])
def test_custom_rate_hooks_respect_independent_group_and_parameter_steps(family, skip_gradients):
    class ScheduledOptimizer(family):
        def get_lr(self, state, group):
            return group['lr'] * (.1 + .03 * state['step'])

    dtype = torch.float32 if family is AdamFP16 else torch.float64
    parameters = [torch.nn.Parameter(torch.tensor([1.2], dtype=dtype)),
                  torch.nn.Parameter(torch.tensor([-.8], dtype=dtype))]
    maximums = [.04, .07]
    coefficient = .2
    kwargs = {'rectify': False} if family is AdaBelief else {}
    optimizer = ScheduledOptimizer([{'params': [parameters[0]], 'lr': maximums[0]},
                                    {'params': [parameters[1]], 'lr': maximums[1]}],
                                   weight_decay=WeightDecay(weight_decay=coefficient), **kwargs)
    expected = [parameter.detach().clone() for parameter in parameters]
    counts = [0, 0]
    for iteration in range(10):
        parameters[0].grad = torch.zeros_like(parameters[0])
        active_second = not skip_gradients or iteration % 2 == 0
        parameters[1].grad = torch.zeros_like(parameters[1]) if active_second else None
        optimizer.step()
        for index, active in enumerate((True, active_second)):
            if active:
                counts[index] += 1
                rate = maximums[index] * (.1 + .03 * counts[index])
                expected[index] *= 1. - rate * coefficient
            tolerance = 3e-7 if dtype == torch.float32 else 2e-12
            torch.testing.assert_close(parameters[index], expected[index], rtol=tolerance, atol=tolerance)
            assert optimizer.state[parameters[index]]['step'] == counts[index]
            assert optimizer.param_groups[index]['lr'] == maximums[index]
