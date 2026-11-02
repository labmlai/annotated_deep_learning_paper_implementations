"""Regressions for the conditional BERT corruption distribution."""

import math

import pytest
import torch

from labml_nn.transformers.mlm import MLM


def make_mlm(**kwargs):
    return MLM(padding_token=0, mask_token=1, no_mask_tokens=[2], n_tokens=1000,
               masking_prob=1, **kwargs)


@pytest.mark.parametrize('unchanged,random', [(0.1, 0.1), (0.2, 0.3), (1.0, 0.0),
                                           (0.0, 1.0), (0.0, 0.0)])
def test_mlm_corruption_outcomes_are_disjoint_and_have_configured_mass(monkeypatch, unchanged, random):
    x = torch.full((100, 1), 3, dtype=torch.long)
    probability_grid = torch.arange(100, dtype=torch.float64).reshape(100, 1) / 100
    draws = iter([torch.zeros_like(probability_grid), probability_grid, probability_grid])
    monkeypatch.setattr(torch, 'rand', lambda *args, **kwargs: next(draws).clone())
    monkeypatch.setattr(torch, 'randint', lambda low, high, size, **kwargs: torch.full(size, 4, dtype=torch.long))
    mlm = make_mlm(no_change_prob=unchanged, randomize_prob=random)

    masked, labels = mlm(x)

    assert masked is x
    assert int((masked == 3).sum()) == round(100 * unchanged)
    assert int((masked == 4).sum()) == round(100 * random)
    assert int((masked == 1).sum()) == round(100 * (1 - unchanged - random))
    assert torch.equal(labels, torch.full_like(labels, 3))


def test_mlm_special_tokens_and_loss_targets_are_preserved():
    x = torch.tensor([[0, 1, 2, 3], [2, 3, 0, 1]])
    original = x.clone()
    mlm = make_mlm(no_change_prob=0, randomize_prob=0)

    masked, labels = mlm(x)

    expected_input = original.clone()
    expected_input[original == 3] = 1
    expected_labels = torch.zeros_like(original)
    expected_labels[original == 3] = 3
    assert torch.equal(masked, expected_input)
    assert torch.equal(labels, expected_labels)
    assert torch.equal(original, torch.tensor([[0, 1, 2, 3], [2, 3, 0, 1]]))


def test_mlm_conditional_masses_with_native_random_draws():
    torch.manual_seed(781)
    x = torch.full((100000, 1), 3, dtype=torch.long)
    mlm = make_mlm(no_change_prob=0.2, randomize_prob=0.3)

    masked, labels = mlm(x)

    # Random replacements can also be the mask/original token with probability
    # 1/n_tokens; include that probability in the observable reference masses.
    expected = {1: 0.5 + 0.3 / 1000, 3: 0.2 + 0.3 / 1000}
    for token, probability in expected.items():
        count = int((masked == token).sum())
        tolerance = 6 * math.sqrt(masked.numel() * probability * (1 - probability))
        assert abs(count - masked.numel() * probability) < tolerance
    assert torch.equal(labels, torch.full_like(labels, 3))


@pytest.mark.parametrize('kwargs', [{'randomize_prob': 0.6, 'no_change_prob': 0.5},
                                   {'randomize_prob': -0.1}, {'no_change_prob': 1.1},
                                   {'no_change_prob': float('nan')}])
def test_mlm_rejects_invalid_corruption_distributions(kwargs):
    with pytest.raises(ValueError):
        make_mlm(**kwargs)
