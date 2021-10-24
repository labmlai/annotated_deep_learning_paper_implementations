"""
---
title: "Evidential Deep Learning to Quantify Classification Uncertainty"
summary: >
 A PyTorch implementation/tutorial of the paper Evidential Deep Learning to Quantify Classification
 Uncertainty.
---

# Evidential Deep Learning to Quantify Classification Uncertainty

This is a [PyTorch](https://pytorch.org) implementation of the paper
[Evidential Deep Learning to Quantify Classification Uncertainty](https://papers.labml.ai/paper/1806.01768).

[Dampster-Shafer Theory of Evidence](https://en.wikipedia.org/wiki/Dempster%E2%80%93Shafer_theory)
assigns belief masses a set of classes (unlike assigning a probability to a single class).
Sum of the masses of all subsets is $1$.
Individual class probabilities (plausibilities) can be derived from these masses.

Assigning a mass to the set of all classes means it can be any one of the classes; i.e. saying "I don't know".

If there are $K$ classes, we assign masses $b_k \ge 0$ to each of the classes and
 an overall uncertainty mass $u \ge 0$ to all classes.

$$u + \sum_{k=1}^K b_k = 1$$

Belief masses $b_k$ and $u$ can be computed from evidence $e_k \ge 0$, as $b_k = \frac{e_k}{S}$
and $u = \frac{K}{S}$ where $S = \sum_{k=1}^K (e_k + 1)$.
Paper uses term evidence as a measure of the amount of support
collected from data in favor of a sample to be classified into a certain class.

This corresponds to a [Dirichlet distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution)
with parameters $\textcolor{orange}{\alpha_k} = e_k + 1$, and
 $\textcolor{orange}{\alpha_0} = S = \sum_{k=1}^K \textcolor{orange}{\alpha_k}$ is known as the Dirichlet strength.
Dirichlet distribution $D(\mathbf{p} \vert \textcolor{orange}{\mathbf{\alpha}})$
 is a distribution over categorical distribution; i.e. you can sample class probabilities
from a Dirichlet distribution.
The expected probability for class $k$ is $\hat{p}_k = \frac{\textcolor{orange}{\alpha_k}}{S}$.

We get the model to output evidences
$$\mathbf{e} = \textcolor{orange}{\mathbf{\alpha}} - 1 = f(\mathbf{x} | \Theta)$$
 for a given input $\mathbf{x}$.
We use a function such as
 [ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html) or a
 [Softplus](https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html)
 at the final layer to get $f(\mathbf{x} | \Theta) \ge 0$.

The paper proposes a few loss functions to train the model, which we have implemented below.

Here is the [training code `experiment.py`](experiment.html) to train a model on MNIST dataset.

[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/f82b2bfc01ba11ecbb2aa16a33570106)
"""

import torch

from labml import tracker
from labml_helpers.module import Module


class MaximumLikelihoodLoss(Module):
    """
    <a id="MaximumLikelihoodLoss"></a>

    ## Type II Maximum Likelihood Loss

    The distribution $D(\mathbf{p} \vert \textcolor{orange}{\mathbf{\alpha}})$ is a prior on the likelihood
    $Multi(\mathbf{y} \vert p)$,
     and the negative log marginal likelihood is calculated by integrating over class probabilities
     $\mathbf{p}$.

    If target probabilities (one-hot targets) are $y_k$ for a given sample the loss is,

    \begin{align}
    \mathcal{L}(\Theta)
    &= -\log \Bigg(
     \int
      \prod_{k=1}^K p_k^{y_k}
      \frac{1}{B(\textcolor{orange}{\mathbf{\alpha}})}
      \prod_{k=1}^K p_k^{\textcolor{orange}{\alpha_k} - 1}
     d\mathbf{p}
     \Bigg ) \\
    &= \sum_{k=1}^K y_k \bigg( \log S - \log \textcolor{orange}{\alpha_k} \bigg)
    \end{align}
    """
    def forward(self, evidence: torch.Tensor, target: torch.Tensor):
        """
        * `evidence` is $\mathbf{e} \ge 0$ with shape `[batch_size, n_classes]`
        * `target` is $\mathbf{y}$ with shape `[batch_size, n_classes]`
        """
        # $\textcolor{orange}{\alpha_k} = e_k + 1$
        alpha = evidence + 1.
        # $S = \sum_{k=1}^K \textcolor{orange}{\alpha_k}$
        strength = alpha.sum(dim=-1)

        # Losses $\mathcal{L}(\Theta) = \sum_{k=1}^K y_k \bigg( \log S - \log \textcolor{orange}{\alpha_k} \bigg)$
        loss = (target * (strength.log()[:, None] - alpha.log())).sum(dim=-1)

        # Mean loss over the batch
        return loss.mean()


class CrossEntropyBayesRisk(Module):
    """
    <a id="CrossEntropyBayesRisk"></a>

    ## Bayes Risk with Cross Entropy Loss

    Bayes risk is the overall maximum cost of making incorrect estimates.
    It takes a cost function that gives the cost of making an incorrect estimate
    and sums it over all possible outcomes based on probability distribution.

    Here the cost function is cross-entropy loss, for one-hot coded $\mathbf{y}$
    $$\sum_{k=1}^K -y_k \log p_k$$

    We integrate this cost over all $\mathbf{p}$

    \begin{align}
    \mathcal{L}(\Theta)
    &= -\log \Bigg(
     \int
      \Big[ \sum_{k=1}^K -y_k \log p_k \Big]
      \frac{1}{B(\textcolor{orange}{\mathbf{\alpha}})}
      \prod_{k=1}^K p_k^{\textcolor{orange}{\alpha_k} - 1}
     d\mathbf{p}
     \Bigg ) \\
    &= \sum_{k=1}^K y_k \bigg( \psi(S) - \psi( \textcolor{orange}{\alpha_k} ) \bigg)
    \end{align}

    where $\psi(\cdot)$ is the $digamma$ function.
    """

    def forward(self, evidence: torch.Tensor, target: torch.Tensor):
        """
        * `evidence` is $\mathbf{e} \ge 0$ with shape `[batch_size, n_classes]`
        * `target` is $\mathbf{y}$ with shape `[batch_size, n_classes]`
        """
        # $\textcolor{orange}{\alpha_k} = e_k + 1$
        alpha = evidence + 1.
        # $S = \sum_{k=1}^K \textcolor{orange}{\alpha_k}$
        strength = alpha.sum(dim=-1)

        # Losses $\mathcal{L}(\Theta) = \sum_{k=1}^K y_k \bigg( \psi(S) - \psi( \textcolor{orange}{\alpha_k} ) \bigg)$
        loss = (target * (torch.digamma(strength)[:, None] - torch.digamma(alpha))).sum(dim=-1)

        # Mean loss over the batch
        return loss.mean()


class SquaredErrorBayesRisk(Module):
    """
    <a id="SquaredErrorBayesRisk"></a>

    ## Bayes Risk with Squared Error Loss

    Here the cost function is squared error,
    $$\sum_{k=1}^K (y_k - p_k)^2 = \Vert \mathbf{y} - \mathbf{p} \Vert_2^2$$

    We integrate this cost over all $\mathbf{p}$

    \begin{align}
    \mathcal{L}(\Theta)
    &= -\log \Bigg(
     \int
      \Big[ \sum_{k=1}^K (y_k - p_k)^2 \Big]
      \frac{1}{B(\textcolor{orange}{\mathbf{\alpha}})}
      \prod_{k=1}^K p_k^{\textcolor{orange}{\alpha_k} - 1}
     d\mathbf{p}
     \Bigg ) \\
    &= \sum_{k=1}^K \mathbb{E} \Big[ y_k^2 -2 y_k p_k + p_k^2 \Big] \\
    &= \sum_{k=1}^K \Big( y_k^2 -2 y_k \mathbb{E}[p_k] + \mathbb{E}[p_k^2] \Big)
    \end{align}

    Where $$\mathbb{E}[p_k] = \hat{p}_k = \frac{\textcolor{orange}{\alpha_k}}{S}$$
    is the expected probability when sampled from the Dirichlet distribution
    and $$\mathbb{E}[p_k^2] = \mathbb{E}[p_k]^2 + \text{Var}(p_k)$$
     where
    $$\text{Var}(p_k) = \frac{\textcolor{orange}{\alpha_k}(S - \textcolor{orange}{\alpha_k})}{S^2 (S + 1)}
    = \frac{\hat{p}_k(1 - \hat{p}_k)}{S + 1}$$
     is the variance.

    This gives,

    \begin{align}
    \mathcal{L}(\Theta)
    &= \sum_{k=1}^K \Big( y_k^2 -2 y_k \mathbb{E}[p_k] + \mathbb{E}[p_k^2] \Big) \\
    &= \sum_{k=1}^K \Big( y_k^2 -2 y_k \mathbb{E}[p_k] +  \mathbb{E}[p_k]^2 + \text{Var}(p_k) \Big) \\
    &= \sum_{k=1}^K \Big( \big( y_k -\mathbb{E}[p_k] \big)^2 + \text{Var}(p_k) \Big) \\
    &= \sum_{k=1}^K \Big( ( y_k -\hat{p}_k)^2 + \frac{\hat{p}_k(1 - \hat{p}_k)}{S + 1} \Big)
    \end{align}

    This first part of the equation $\big(y_k -\mathbb{E}[p_k]\big)^2$ is the error term and
    the second part is the variance.
    """

    def forward(self, evidence: torch.Tensor, target: torch.Tensor):
        """
        * `evidence` is $\mathbf{e} \ge 0$ with shape `[batch_size, n_classes]`
        * `target` is $\mathbf{y}$ with shape `[batch_size, n_classes]`
        """
        # $\textcolor{orange}{\alpha_k} = e_k + 1$
        alpha = evidence + 1.
        # $S = \sum_{k=1}^K \textcolor{orange}{\alpha_k}$
        strength = alpha.sum(dim=-1)
        # $\hat{p}_k = \frac{\textcolor{orange}{\alpha_k}}{S}$
        p = alpha / strength[:, None]

        # Error $(y_k -\hat{p}_k)^2$
        err = (target - p) ** 2
        # Variance $\text{Var}(p_k) = \frac{\hat{p}_k(1 - \hat{p}_k)}{S + 1}$
        var = p * (1 - p) / (strength[:, None] + 1)

        # Sum of them
        loss = (err + var).sum(dim=-1)

        # Mean loss over the batch
        return loss.mean()


class KLDivergenceLoss(Module):
    """
    <a id="KLDivergenceLoss"></a>

    ## KL Divergence Regularization Loss

    This tries to shrink the total evidence to zero if the sample cannot be correctly classified.

    First we calculate $\tilde{\alpha}_k = y_k + (1 - y_k) \textcolor{orange}{\alpha_k}$ the
    Dirichlet parameters after remove the correct evidence.

    \begin{align}
    &KL \Big[ D(\mathbf{p} \vert \mathbf{\tilde{\alpha}}) \Big \Vert
    D(\mathbf{p} \vert <1, \dots, 1>\Big] \\
    &= \log \Bigg( \frac{\Gamma \Big( \sum_{k=1}^K \tilde{\alpha}_k \Big)}
    {\Gamma(K) \prod_{k=1}^K \Gamma(\tilde{\alpha}_k)} \Bigg)
    + \sum_{k=1}^K (\tilde{\alpha}_k - 1)
    \Big[ \psi(\tilde{\alpha}_k) - \psi(\tilde{S}) \Big]
    \end{align}

    where $\Gamma(\cdot)$ is the gamma function,
    $\psi(\cdot)$ is the $digamma$ function and
    $\tilde{S} = \sum_{k=1}^K \tilde{\alpha}_k$
    """
    def forward(self, evidence: torch.Tensor, target: torch.Tensor):
        """
        * `evidence` is $\mathbf{e} \ge 0$ with shape `[batch_size, n_classes]`
        * `target` is $\mathbf{y}$ with shape `[batch_size, n_classes]`
        """
        # $\textcolor{orange}{\alpha_k} = e_k + 1$
        alpha = evidence + 1.
        # Number of classes
        n_classes = evidence.shape[-1]
        # Remove non-misleading evidence
        # $$\tilde{\alpha}_k = y_k + (1 - y_k) \textcolor{orange}{\alpha_k}$$
        alpha_tilde = target + (1 - target) * alpha
        # $\tilde{S} = \sum_{k=1}^K \tilde{\alpha}_k$
        strength_tilde = alpha_tilde.sum(dim=-1)

        # The first term
        #
        # \begin{align}
        # &\log \Bigg( \frac{\Gamma \Big( \sum_{k=1}^K \tilde{\alpha}_k \Big)}
        #     {\Gamma(K) \prod_{k=1}^K \Gamma(\tilde{\alpha}_k)} \Bigg) \\
        # &= \log \Gamma \Big( \sum_{k=1}^K \tilde{\alpha}_k \Big)
        #   - \log \Gamma(K)
        #   - \sum_{k=1}^K \log \Gamma(\tilde{\alpha}_k)
        # \end{align}
        first = (torch.lgamma(alpha_tilde.sum(dim=-1))
                 - torch.lgamma(alpha_tilde.new_tensor(float(n_classes)))
                 - (torch.lgamma(alpha_tilde)).sum(dim=-1))

        # The second term
        # $$\sum_{k=1}^K (\tilde{\alpha}_k - 1)
        #     \Big[ \psi(\tilde{\alpha}_k) - \psi(\tilde{S}) \Big]$$
        second = (
                (alpha_tilde - 1) *
                (torch.digamma(alpha_tilde) - torch.digamma(strength_tilde)[:, None])
        ).sum(dim=-1)

        # Sum of the terms
        loss = first + second

        # Mean loss over the batch
        return loss.mean()


class TrackStatistics(Module):
    """
    <a id="TrackStatistics"></a>

    ### Track statistics

    This module computes statistics and tracks them with [labml `tracker`](https://docs.labml.ai/api/tracker.html).
    """
    def forward(self, evidence: torch.Tensor, target: torch.Tensor):
        # Number of classes
        n_classes = evidence.shape[-1]
        # Predictions that correctly match with the target (greedy sampling based on highest probability)
        match = evidence.argmax(dim=-1).eq(target.argmax(dim=-1))
        # Track accuracy
        tracker.add('accuracy.', match.sum() / match.shape[0])

        # $\textcolor{orange}{\alpha_k} = e_k + 1$
        alpha = evidence + 1.
        # $S = \sum_{k=1}^K \textcolor{orange}{\alpha_k}$
        strength = alpha.sum(dim=-1)

        # $\hat{p}_k = \frac{\textcolor{orange}{\alpha_k}}{S}$
        expected_probability = alpha / strength[:, None]
        # Expected probability of the selected (greedy highset probability) class
        expected_probability, _ = expected_probability.max(dim=-1)

        # Uncertainty mass $u = \frac{K}{S}$
        uncertainty_mass = n_classes / strength

        # Track $u$ for correctly predictions
        tracker.add('u.succ.', uncertainty_mass.masked_select(match))
        # Track $u$ for incorrect predictions
        tracker.add('u.fail.', uncertainty_mass.masked_select(~match))
        # Track $\hat{p}_k$ for correctly predictions
        tracker.add('prob.succ.', expected_probability.masked_select(match))
        # Track $\hat{p}_k$ for incorrect predictions
        tracker.add('prob.fail.', expected_probability.masked_select(~match))
