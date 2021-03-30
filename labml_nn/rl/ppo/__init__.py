"""
---
title: Proximal Policy Optimization - PPO
summary: >
 An annotated implementation of Proximal Policy Optimization - PPO algorithm in PyTorch.
---

# Proximal Policy Optimization - PPO

This is a [PyTorch](https://pytorch.org) implementation of
[Proximal Policy Optimization - PPO](https://arxiv.org/abs/1707.06347).

PPO is a policy gradient method for reinforcement learning.
Simple policy gradient methods do a single gradient update per sample (or a set of samples).
Doing multiple gradient steps for a single sample causes problems
because the policy deviates too much, producing a bad policy.
PPO lets us do multiple gradient updates per sample by trying to keep the
policy close to the policy that was used to sample data.
It does so by clipping gradient flow if the updated policy
is not close to the policy used to sample the data.

You can find an experiment that uses it [here](experiment.html).
The experiment uses [Generalized Advantage Estimation](gae.html).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lab-ml/nn/blob/master/labml_nn/rl/ppo/experiment.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/6eff28a0910e11eb9b008db315936e2f)
"""

import torch

from labml_helpers.module import Module
from labml_nn.rl.ppo.gae import GAE


class ClippedPPOLoss(Module):
    """
    ## PPO Loss

    Here's how the PPO update rule is derived.

    We want to maximize policy reward
     $$\max_\theta J(\pi_\theta) =
       \mathop{\mathbb{E}}_{\tau \sim \pi_\theta}\Biggl[\sum_{t=0}^\infty \gamma^t r_t \Biggr]$$
     where $r$ is the reward, $\pi$ is the policy, $\tau$ is a trajectory sampled from policy,
     and $\gamma$ is the discount factor between $[0, 1]$.

    \begin{align}
    \mathbb{E}_{\tau \sim \pi_\theta} \Biggl[
     \sum_{t=0}^\infty \gamma^t A^{\pi_{OLD}}(s_t, a_t)
    \Biggr] &=
    \\
    \mathbb{E}_{\tau \sim \pi_\theta} \Biggl[
      \sum_{t=0}^\infty \gamma^t \Bigl(
       Q^{\pi_{OLD}}(s_t, a_t) - V^{\pi_{OLD}}(s_t)
      \Bigr)
     \Biggr] &=
    \\
    \mathbb{E}_{\tau \sim \pi_\theta} \Biggl[
      \sum_{t=0}^\infty \gamma^t \Bigl(
       r_t + V^{\pi_{OLD}}(s_{t+1}) - V^{\pi_{OLD}}(s_t)
      \Bigr)
     \Biggr] &=
    \\
    \mathbb{E}_{\tau \sim \pi_\theta} \Biggl[
      \sum_{t=0}^\infty \gamma^t \Bigl(
       r_t
      \Bigr)
     \Biggr]
     - \mathbb{E}_{\tau \sim \pi_\theta}
        \Biggl[V^{\pi_{OLD}}(s_0)\Biggr] &=
    J(\pi_\theta) - J(\pi_{\theta_{OLD}})
    \end{align}

    So,
     $$\max_\theta J(\pi_\theta) =
       \max_\theta \mathbb{E}_{\tau \sim \pi_\theta} \Biggl[
          \sum_{t=0}^\infty \gamma^t A^{\pi_{OLD}}(s_t, a_t)
       \Biggr]$$

    Define discounted-future state distribution,
     $$d^\pi(s) = (1 - \gamma) \sum_{t=0}^\infty \gamma^t P(s_t = s | \pi)$$

    Then,
    \begin{align}
    J(\pi_\theta) - J(\pi_{\theta_{OLD}})
    &= \mathbb{E}_{\tau \sim \pi_\theta} \Biggl[
     \sum_{t=0}^\infty \gamma^t A^{\pi_{OLD}}(s_t, a_t)
    \Biggr]
    \\
    &= \frac{1}{1 - \gamma}
     \mathbb{E}_{s \sim d^{\pi_\theta}, a \sim \pi_\theta} \Bigl[
      A^{\pi_{OLD}}(s, a)
     \Bigr]
    \end{align}

    Importance sampling $a$ from $\pi_{\theta_{OLD}}$,

    \begin{align}
    J(\pi_\theta) - J(\pi_{\theta_{OLD}})
    &= \frac{1}{1 - \gamma}
     \mathbb{E}_{s \sim d^{\pi_\theta}, a \sim \pi_\theta} \Bigl[
      A^{\pi_{OLD}}(s, a)
     \Bigr]
    \\
    &= \frac{1}{1 - \gamma}
     \mathbb{E}_{s \sim d^{\pi_\theta}, a \sim \pi_{\theta_{OLD}}} \Biggl[
      \frac{\pi_\theta(a|s)}{\pi_{\theta_{OLD}}(a|s)} A^{\pi_{OLD}}(s, a)
     \Biggr]
    \end{align}

    Then we assume $d^\pi_\theta(s)$ and  $d^\pi_{\theta_{OLD}}(s)$ are similar.
    The error we introduce to $J(\pi_\theta) - J(\pi_{\theta_{OLD}})$
     by this assumption is bound by the KL divergence between
     $\pi_\theta$ and $\pi_{\theta_{OLD}}$.
    [Constrained Policy Optimization](https://arxiv.org/abs/1705.10528)
     shows the proof of this. I haven't read it.


    \begin{align}
    J(\pi_\theta) - J(\pi_{\theta_{OLD}})
    &= \frac{1}{1 - \gamma}
     \mathop{\mathbb{E}}_{s \sim d^{\pi_\theta} \atop a \sim \pi_{\theta_{OLD}}} \Biggl[
      \frac{\pi_\theta(a|s)}{\pi_{\theta_{OLD}}(a|s)} A^{\pi_{OLD}}(s, a)
     \Biggr]
    \\
    &\approx \frac{1}{1 - \gamma}
     \mathop{\mathbb{E}}_{\color{orange}{s \sim d^{\pi_{\theta_{OLD}}}}
     \atop a \sim \pi_{\theta_{OLD}}} \Biggl[
      \frac{\pi_\theta(a|s)}{\pi_{\theta_{OLD}}(a|s)} A^{\pi_{OLD}}(s, a)
     \Biggr]
    \\
    &= \frac{1}{1 - \gamma} \mathcal{L}^{CPI}
    \end{align}
    """

    def __init__(self):
        super().__init__()

    def __call__(self, log_pi: torch.Tensor, sampled_log_pi: torch.Tensor,
                 advantage: torch.Tensor, clip: float) -> torch.Tensor:
        # ratio $r_t(\theta) = \frac{\pi_\theta (a_t|s_t)}{\pi_{\theta_{OLD}} (a_t|s_t)}$;
        # *this is different from rewards* $r_t$.
        ratio = torch.exp(log_pi - sampled_log_pi)

        # ### Cliping the policy ratio
        #
        # \begin{align}
        # \mathcal{L}^{CLIP}(\theta) =
        #  \mathbb{E}_{a_t, s_t \sim \pi_{\theta{OLD}}} \biggl[
        #    min \Bigl(r_t(\theta) \bar{A_t},
        #              clip \bigl(
        #               r_t(\theta), 1 - \epsilon, 1 + \epsilon
        #              \bigr) \bar{A_t}
        #    \Bigr)
        #  \biggr]
        # \end{align}
        #
        # The ratio is clipped to be close to 1.
        # We take the minimum so that the gradient will only pull
        # $\pi_\theta$ towards $\pi_{\theta_{OLD}}$ if the ratio is
        # not between $1 - \epsilon$ and $1 + \epsilon$.
        # This keeps the KL divergence between $\pi_\theta$
        #  and $\pi_{\theta_{OLD}}$ constrained.
        # Large deviation can cause performance collapse;
        #  where the policy performance drops and doesn't recover because
        #  we are sampling from a bad policy.
        #
        # Using the normalized advantage
        #  $\bar{A_t} = \frac{\hat{A_t} - \mu(\hat{A_t})}{\sigma(\hat{A_t})}$
        #  introduces a bias to the policy gradient estimator,
        #  but it reduces variance a lot.
        clipped_ratio = ratio.clamp(min=1.0 - clip,
                                    max=1.0 + clip)
        policy_reward = torch.min(ratio * advantage,
                                  clipped_ratio * advantage)

        self.clip_fraction = (abs((ratio - 1.0)) > clip).to(torch.float).mean()

        return -policy_reward.mean()


class ClippedValueFunctionLoss(Module):
    """
    ## Clipped Value Function Loss

    Similarly we clip the value function update also.

    \begin{align}
    V^{\pi_\theta}_{CLIP}(s_t)
     &= clip\Bigl(V^{\pi_\theta}(s_t) - \hat{V_t}, -\epsilon, +\epsilon\Bigr)
    \\
    \mathcal{L}^{VF}(\theta)
     &= \frac{1}{2} \mathbb{E} \biggl[
      max\Bigl(\bigl(V^{\pi_\theta}(s_t) - R_t\bigr)^2,
          \bigl(V^{\pi_\theta}_{CLIP}(s_t) - R_t\bigr)^2\Bigr)
     \biggr]
    \end{align}

    Clipping makes sure the value function $V_\theta$ doesn't deviate
     significantly from $V_{\theta_{OLD}}$.

    """
    def __call__(self, value: torch.Tensor, sampled_value: torch.Tensor, sampled_return: torch.Tensor, clip: float):
        clipped_value = sampled_value + (value - sampled_value).clamp(min=-clip, max=clip)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        return 0.5 * vf_loss.mean()
