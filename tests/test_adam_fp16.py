"""Checkpoint continuity tests for the optimizer's FP32 master state."""

import copy
import io

import pytest
import torch

from labml_nn.optimizers import WeightDecay
from labml_nn.optimizers.adam_fp16 import AdamFP16


STATE_KEYS = ('exp_avg', 'exp_avg_sq', 'fp32_copy')


def step(optimizer, parameters, gradients):
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = torch.tensor(gradient, dtype=parameter.dtype)
    optimizer.step()


def serialized_state(optimizer):
    checkpoint = io.BytesIO()
    torch.save(optimizer.state_dict(), checkpoint)
    checkpoint.seek(0)
    return torch.load(checkpoint, weights_only=True)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('optimized_update', [False, True])
@pytest.mark.parametrize('decoupled_decay', [False, True])
def test_adam_fp16_checkpoint_preserves_training_trajectory(dtype, optimized_update, decoupled_decay):
    parameters = [torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=dtype)),
                  torch.nn.Parameter(torch.tensor([-1.0], dtype=dtype))]
    decay = WeightDecay(weight_decay=0.03, weight_decouple=decoupled_decay)
    groups = [{'params': [parameters[0]], 'lr': 0.001},
              {'params': [parameters[1]], 'lr': 0.002}]
    optimizer = AdamFP16(groups, optimized_update=optimized_update, weight_decay=decay)
    gradients = [[0.25, -0.5], [0.125]]
    for _ in range(3):
        step(optimizer, parameters, gradients)
    checkpoint = serialized_state(optimizer)
    restored_parameters = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    restored_groups = [{'params': [restored_parameters[0]]}, {'params': [restored_parameters[1]]}]
    restored = AdamFP16(restored_groups, optimized_update=optimized_update, weight_decay=decay)

    restored.load_state_dict(checkpoint)

    for original, resumed in zip(parameters, restored_parameters):
        for key in STATE_KEYS:
            assert restored.state[resumed][key].dtype == torch.float32
            torch.testing.assert_close(restored.state[resumed][key], optimizer.state[original][key], rtol=0, atol=0)
        assert restored.state[resumed]['step'] == optimizer.state[original]['step']
    assert [g['lr'] for g in restored.param_groups] == [0.001, 0.002]
    for _ in range(12):
        step(optimizer, parameters, gradients)
        step(restored, restored_parameters, gradients)
    for original, resumed in zip(parameters, restored_parameters):
        torch.testing.assert_close(resumed, original, rtol=0, atol=0)
        for key in STATE_KEYS:
            torch.testing.assert_close(restored.state[resumed][key], optimizer.state[original][key], rtol=0, atol=0)


def test_adam_fp16_loading_uninitialized_and_partially_initialized_parameters():
    parameters = [torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16)) for _ in range(2)]
    optimizer = AdamFP16(parameters)
    optimizer.load_state_dict(serialized_state(optimizer))
    assert len(optimizer.state) == 0
    parameters[0].grad = torch.tensor([0.25, 0.5], dtype=torch.bfloat16)
    optimizer.step()
    checkpoint = serialized_state(optimizer)
    resumed_parameters = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    resumed = AdamFP16(resumed_parameters)

    resumed.load_state_dict(checkpoint)

    assert resumed_parameters[1] not in resumed.state
    for key in STATE_KEYS:
        assert resumed.state[resumed_parameters[0]][key].dtype == torch.float32
        torch.testing.assert_close(resumed.state[resumed_parameters[0]][key], optimizer.state[parameters[0]][key], rtol=0, atol=0)


def test_adam_fp16_state_load_respects_pre_and_post_hooks():
    if not hasattr(AdamFP16, 'register_load_state_dict_pre_hook'):
        pytest.skip('Optimizer load hooks are unavailable in this PyTorch version')
    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    optimizer = AdamFP16([parameter])
    step(optimizer, [parameter], [[0.25, -0.5]])
    checkpoint = serialized_state(optimizer)
    checkpoint_before = copy.deepcopy(checkpoint)
    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed = AdamFP16([resumed_parameter])
    observed = []

    def adapt(optimizer, state_dict):
        state_dict = copy.deepcopy(state_dict)
        state_dict['state'][0]['fp32_copy'].add_(0.000125)
        return state_dict

    def inspect(optimizer):
        observed.append(optimizer.state[resumed_parameter]['fp32_copy'].clone())

    resumed.register_load_state_dict_pre_hook(adapt)
    resumed.register_load_state_dict_post_hook(inspect)
    resumed.load_state_dict(checkpoint)

    expected = checkpoint['state'][0]['fp32_copy'] + 0.000125
    torch.testing.assert_close(observed[0], expected, rtol=0, atol=0)
    assert observed[0].dtype == torch.float32
    assert len(resumed._optimizer_load_state_dict_pre_hooks) == 1
    assert len(resumed._optimizer_load_state_dict_post_hooks) == 1
    for key in STATE_KEYS:
        torch.testing.assert_close(checkpoint['state'][0][key], checkpoint_before['state'][0][key], rtol=0, atol=0)


def test_adam_fp16_invalid_checkpoint_keeps_native_validation_and_removes_hooks():
    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    optimizer = AdamFP16([parameter])
    checkpoint = optimizer.state_dict()
    checkpoint['param_groups'][0]['params'].append(99)

    with pytest.raises(ValueError, match="doesn't match the size"):
        optimizer.load_state_dict(checkpoint)

    if hasattr(optimizer, '_optimizer_load_state_dict_pre_hooks'):
        assert not optimizer._optimizer_load_state_dict_pre_hooks
        assert not optimizer._optimizer_load_state_dict_post_hooks


def test_adam_fp16_loading_without_public_load_hook_api(monkeypatch):
    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    optimizer = AdamFP16([parameter])
    step(optimizer, [parameter], [[0.25, -0.5]])
    checkpoint = serialized_state(optimizer)
    if hasattr(torch.optim.Optimizer, 'register_load_state_dict_pre_hook'):
        monkeypatch.delattr(torch.optim.Optimizer, 'register_load_state_dict_pre_hook')
    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed = AdamFP16([resumed_parameter])

    resumed.load_state_dict(checkpoint)

    for key in STATE_KEYS:
        assert resumed.state[resumed_parameter][key].dtype == torch.float32
        torch.testing.assert_close(resumed.state[resumed_parameter][key], checkpoint['state'][0][key], rtol=0, atol=0)


def test_adam_fp16_loading_does_not_alias_the_checkpoint_tensors():
    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    optimizer = AdamFP16([parameter])
    step(optimizer, [parameter], [[0.25, -0.5]])
    checkpoint = serialized_state(optimizer)
    original = copy.deepcopy(checkpoint)
    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed = AdamFP16([resumed_parameter])

    resumed.load_state_dict(checkpoint)
    step(resumed, [resumed_parameter], [[0.5, 0.25]])

    for key in STATE_KEYS:
        torch.testing.assert_close(checkpoint['state'][0][key], original['state'][0][key], rtol=0, atol=0)
