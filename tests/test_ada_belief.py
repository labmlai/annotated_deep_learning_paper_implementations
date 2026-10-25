"""AdaBelief configuration must select the requested optimizer recurrence."""
import math

import pytest
import torch

from labml_nn.optimizers.ada_belief import AdaBelief


@pytest.mark.parametrize('rectify', [False, True])
@pytest.mark.parametrize('amsgrad', [False, True])
@pytest.mark.parametrize('degenerate_to_sgd', [False, True])
def test_ada_belief_trajectory(rectify, amsgrad, degenerate_to_sgd):
    initial = torch.tensor([1.0, -2.0], dtype=torch.float64)
    parameter = torch.nn.Parameter(initial.clone())
    lr, eps = 0.1, 0.03
    beta1, beta2 = 0.6, 0.8
    optimizer = AdaBelief([parameter], lr=lr, eps=eps, betas=(beta1, beta2),
                          rectify=rectify, amsgrad=amsgrad,
                          degenerate_to_sgd=degenerate_to_sgd)
    expected = initial.clone()
    momentum = torch.zeros_like(initial)
    variance = torch.zeros_like(initial)
    maximum = torch.zeros_like(initial)
    # Large early residuals and smaller later residuals distinguish AMSGrad.
    gradients = [torch.tensor(g, dtype=torch.float64) for g in
                 [(3.0, -2.0), (-1.5, 4.0), (0.01, -0.03), (0.0, 0.0)]]
    for step in range(1, 17):
        gradient = gradients[(step - 1) % len(gradients)]
        parameter.grad = gradient.clone()
        optimizer.step()

        momentum = beta1 * momentum + (1 - beta1) * gradient
        variance = beta2 * variance + (1 - beta2) * (gradient - momentum).square()
        maximum = torch.maximum(maximum, variance)
        selected_variance = maximum if amsgrad else variance
        corrected_momentum = momentum / (1 - beta1 ** step)
        denominator = ((selected_variance + eps) / (1 - beta2 ** step)).sqrt() + eps
        rho_inf = 2 / (1 - beta2) - 1
        rho = rho_inf - 2 * step * beta2 ** step / (1 - beta2 ** step)
        if not rectify:
            expected = expected - lr * corrected_momentum / denominator
        elif rho >= 5:
            factor = math.sqrt((rho - 4) * (rho - 2) * rho_inf /
                               ((rho_inf - 4) * (rho_inf - 2) * rho))
            expected = expected - lr * factor * corrected_momentum / denominator
        elif degenerate_to_sgd:
            expected = expected - lr * corrected_momentum

        torch.testing.assert_close(parameter, expected, rtol=1e-13, atol=1e-13)
        state = optimizer.state[parameter]
        torch.testing.assert_close(state['exp_avg'], momentum)
        torch.testing.assert_close(state['exp_avg_var'], variance)
        assert ('max_exp_avg_var' in state) is amsgrad
        if amsgrad:
            torch.testing.assert_close(state['max_exp_avg_var'], maximum)


def test_ada_belief_default_starts_with_sgd_update():
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    optimizer = AdaBelief([parameter], lr=0.1)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    torch.testing.assert_close(parameter, torch.tensor([0.9], dtype=torch.float64))
    assert 'max_exp_avg_var' not in optimizer.state[parameter]


def test_ada_belief_custom_defaults_reach_parameter_groups():
    class ScaledLearningRateAdaBelief(AdaBelief):
        def get_lr(self, state, group):
            return group['lr'] * group['lr_scale']

    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    optimizer = ScaledLearningRateAdaBelief([parameter], lr=0.1,
                                          defaults={'lr_scale': 0.25})
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    torch.testing.assert_close(parameter, torch.tensor([0.975], dtype=torch.float64))
    assert optimizer.defaults['lr_scale'] == 0.25


def test_ada_belief_parameter_group_amsgrad_override():
    first = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    second = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    optimizer = AdaBelief([{'params': [first], 'amsgrad': True},
                          {'params': [second], 'amsgrad': False}],
                         lr=0.1, degenerate_to_sgd=False)
    for parameter in (first, second):
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    torch.testing.assert_close(first, torch.ones_like(first))
    torch.testing.assert_close(second, torch.ones_like(second))
    assert 'max_exp_avg_var' in optimizer.state[first]
    assert 'max_exp_avg_var' not in optimizer.state[second]


def test_ada_belief_skips_parameters_without_gradients():
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    optimizer = AdaBelief([parameter])
    optimizer.step()
    torch.testing.assert_close(parameter, torch.ones_like(parameter))
    assert parameter not in optimizer.state
