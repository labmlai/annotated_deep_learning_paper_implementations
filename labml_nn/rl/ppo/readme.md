# [Proximal Policy Optimization - PPO](https://nn.labml.ai/rl/ppo/index.html)

This is a [PyTorch](https://pytorch.org) implementation of
[Proximal Policy Optimization - PPO](https://papers.labml.ai/paper/1707.06347).

PPO is a policy gradient method for reinforcement learning.
Simple policy gradient methods one do a single gradient update per sample (or a set of samples).
Doing multiple gradient steps for a singe sample causes problems
because the policy deviates too much producing a bad policy.
PPO lets us do multiple gradient updates per sample by trying to keep the
policy close to the policy that was used to sample data.
It does so by clipping gradient flow if the updated policy
is not close to the policy used to sample the data.

You can find an experiment that uses it [here](https://nn.labml.ai/rl/ppo/experiment.html).
The experiment uses [Generalized Advantage Estimation](https://nn.labml.ai/rl/ppo/gae.html).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/rl/ppo/experiment.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/6eff28a0910e11eb9b008db315936e2f)
