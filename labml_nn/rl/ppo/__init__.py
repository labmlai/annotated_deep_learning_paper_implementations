"""
This is a an implementation of [Proximal Policy Optimization - PPO](https://arxiv.org/abs/1707.06347)
 clipped version for Atari Breakout game on OpenAI Gym.
It runs the game environments on multiple processes to sample efficiently.
Advantages are calculated using [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).

*This is based on my original implementation
[on my blog](http://blog.varunajayasiri.com/ml/ppo_pytorch.html)*.
"""

