import torch

from labml import tracker
from labml_helpers.module import Module


class MaximumLikelihoodLoss(Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, evidence: torch.Tensor, target: torch.Tensor):
        alpha = evidence + 1.
        strength = alpha.sum(dim=-1)

        loss = (target * (strength.log()[:, None] - alpha.log())).sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss


class CrossEntropyBayesRisk(Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, evidence: torch.Tensor, target: torch.Tensor):
        alpha = evidence + 1.
        strength = alpha.sum(dim=-1)

        loss = (target * (torch.digamma(strength)[:, None] - torch.digamma(alpha))).sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss


class SquaredErrorBayesRisk(Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, evidence: torch.Tensor, target: torch.Tensor):
        alpha = evidence + 1.
        strength = alpha.sum(dim=-1)
        p = alpha / strength[:, None]

        err = (target - p) ** 2
        var = p * (1 - p) / (strength[:, None] + 1)

        loss = (err + var).sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss


class KLDivergenceLoss(Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, evidence: torch.Tensor, target: torch.Tensor):
        alpha = evidence + 1.
        n_classes = evidence.shape[-1]
        # Clean
        alpha = target + (1 - target) * alpha

        first = (torch.lgamma(alpha.sum(dim=-1))
                 - torch.lgamma(alpha.new_tensor(float(n_classes)))
                 - (torch.lgamma(alpha)).sum(dim=-1))

        second = (
                (alpha - 1) *
                (torch.digamma(alpha) - torch.digamma(alpha.sum(dim=-1))[:, None])
        ).sum(dim=-1)

        loss = first + second

        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss


class TrackStatistics(Module):
    def __init__(self):
        super().__init__()

    def forward(self, evidence: torch.Tensor, target: torch.Tensor):
        match = evidence.argmax(dim=-1).eq(target.argmax(dim=-1))

        tracker.add('accuracy.', match.sum() / match.shape[0])

        n_classes = evidence.shape[-1]

        alpha = evidence + 1.
        strength = alpha.sum(dim=-1)

        expected_probability = alpha / strength[:, None]
        expected_probability, _ = expected_probability.max(dim=-1)

        uncertainty_mass = n_classes / strength

        tracker.add('u.succ.', uncertainty_mass.masked_select(match))
        tracker.add('u.fail.', uncertainty_mass.masked_select(~match))
        tracker.add('prob.succ.', expected_probability.masked_select(match))
        tracker.add('prob.fail.', expected_probability.masked_select(~match))
