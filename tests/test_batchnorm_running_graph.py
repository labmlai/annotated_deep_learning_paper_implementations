"""Running statistics must not retain completed training graphs."""
import pytest
import torch
import torch.nn.functional as F

from labml_nn.normalization.batch_norm import BatchNorm

SHAPES = [(4, 3), (2, 3, 5), (2, 3, 2, 4), (3, 3, 2, 2, 2)]


def affine_parameters(module):
    if not module.affine:
        return None, None
    return module.scale, module.shift


def output_and_gradients(module, value):
    actual_input = value.clone().requires_grad_()
    reference_input = value.clone().requires_grad_()
    weight, bias = affine_parameters(module)
    actual = module(actual_input)
    expected = F.batch_norm(reference_input, None, None, weight, bias, training=True, eps=module.eps)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    targets = () if weight is None else (weight, bias)
    loss_weights = torch.linspace(-.9, .8, value.numel(), dtype=value.dtype).reshape_as(value)
    reference_grads = torch.autograd.grad((expected * loss_weights).sum(), (reference_input, *targets))
    actual_grads = torch.autograd.grad((actual * loss_weights).sum(), (actual_input, *targets))
    for actual_grad, reference_grad in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('momentum', [0., .2])
@pytest.mark.parametrize('affine', [False, True])
def test_training_running_stats_are_detached_without_changing_batch_gradients(shape, momentum, affine):
    torch.manual_seed(82)
    module = BatchNorm(3, momentum=momentum, affine=affine).double()
    original_mean, original_var = module.exp_mean, module.exp_var
    mean, variance = torch.zeros(3, dtype=torch.float64), torch.ones(3, dtype=torch.float64)
    axes = tuple(i for i in range(len(shape)) if i != 1)
    for step in range(4):
        value = torch.randn(shape, dtype=torch.float64) * .4 + step * .1
        output_and_gradients(module, value)
        mean = (1 - momentum) * mean + momentum * value.mean(axes)
        variance = (1 - momentum) * variance + momentum * value.var(axes, unbiased=False)
        torch.testing.assert_close(module.exp_mean, mean, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(module.exp_var, variance, rtol=1e-10, atol=1e-10)
        assert not module.exp_mean.requires_grad and module.exp_mean.grad_fn is None
        assert not module.exp_var.requires_grad and module.exp_var.grad_fn is None
        assert module.exp_mean is original_mean and module.exp_var is original_var
    # The same registered inference state remains serializable after updates.
    saved = module.state_dict()
    torch.testing.assert_close(saved['exp_mean'], mean)
    torch.testing.assert_close(saved['exp_var'], variance)


@pytest.mark.parametrize('affine', [False, True])
def test_evaluation_backward_after_completed_training_is_independent_of_training_input(affine):
    torch.manual_seed(43)
    module = BatchNorm(3, momentum=.3, affine=affine).double()
    training_input = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    module(training_input).square().sum().backward()
    previous_training_gradient = training_input.grad.clone()
    mean = module.exp_mean.detach().clone()
    variance = module.exp_var.detach().clone()
    module.zero_grad(set_to_none=True)
    module.eval()
    value = torch.randn(2, 3, 4, dtype=torch.float64)
    actual_input = value.clone().requires_grad_()
    reference_input = value.clone().requires_grad_()
    weight, bias = affine_parameters(module)
    reference_weight = None if weight is None else weight.detach().clone().requires_grad_()
    reference_bias = None if bias is None else bias.detach().clone().requires_grad_()
    actual = module(actual_input)
    expected = F.batch_norm(reference_input, mean, variance, reference_weight, reference_bias, training=False, eps=module.eps)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    expected.square().sum().backward()
    # This must not traverse the already completed graph of training_input.
    actual.square().sum().backward()
    torch.testing.assert_close(actual_input.grad, reference_input.grad, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(training_input.grad, previous_training_gradient, rtol=0., atol=0.)
    if affine:
        torch.testing.assert_close(weight.grad, reference_weight.grad, rtol=1e-9, atol=1e-10)
        torch.testing.assert_close(bias.grad, reference_bias.grad, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('affine', [False, True])
def test_untracked_running_statistics_preserve_training_and_eval_gradients(shape, affine):
    torch.manual_seed(98)
    module = BatchNorm(3, affine=affine, track_running_stats=False).double()
    value = torch.randn(shape, dtype=torch.float64) * .5
    output_and_gradients(module, value)
    module.eval()
    output_and_gradients(module, value)
    assert not hasattr(module, 'exp_mean') and not hasattr(module, 'exp_var')
