[![Join Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://join.slack.com/t/labforml/shared_invite/zt-egj9zvq9-Dl3hhZqobexgT7aVKnD14g/)
[![Twitter](https://img.shields.io/twitter/follow/labmlai?style=social)](https://twitter.com/labmlai)

# [labml.ai Deep Learning Paper Implementaion](https://nn.labml.ai/index.html)

This is a collection of simple PyTorch implementations of
neural networks and related algorithms.
These implementations are documented with explanations,

[The website](https://nn.labml.ai/index.html)
renders these as side-by-side formatted notes.
We believe these would help you understand these algorithms better.

![Screenshot](https://github.com/lab-ml/nn/blob/master/images/dqn.png)

We are actively maintaining this repo and adding new 
implementations almost weekly.
[![Twitter](https://img.shields.io/twitter/follow/labmlai?style=social)](https://twitter.com/labmlai) for updates.

## Modules

#### ✨ [Transformers](https://nn.labml.ai/transformers/index.html)

* [Multi-headed attention](https://nn.labml.ai/transformers/mha.html)
* [Transformer building blocks](https://nn.labml.ai/transformers/models.html) 
* [Transformer XL](https://nn.labml.ai/transformers/xl/index.html)
    * [Relative multi-headed attention](https://nn.labml.ai/transformers/xl/relative_mha.html)
* [Compressive Transformer](https://nn.labml.ai/transformers/compressive/index.html)
* [GPT Architecture](https://nn.labml.ai/transformers/gpt/index.html)
* [GLU Variants](https://nn.labml.ai/transformers/glu_variants/simple.html)
* [kNN-LM: Generalization through Memorization](https://nn.labml.ai/transformers/knn)
* [Feedback Transformer](https://nn.labml.ai/transformers/feedback/index.html)
* [Switch Transformer](https://nn.labml.ai/transformers/switch/index.html)
* [Fast Weights Transformer](https://nn.labml.ai/transformers/fast_weights/index.html)
* [FNet](https://nn.labml.ai/transformers/fnet/index.html)
* [Attention Free Transformer](https://nn.labml.ai/transformers/aft/index.html)
* [Masked Language Model](https://nn.labml.ai/transformers/mlm/index.html)
* [MLP-Mixer: An all-MLP Architecture for Vision](https://nn.labml.ai/transformers/mlp_mixer/index.html)
* [Pay Attention to MLPs (gMLP)](https://nn.labml.ai/transformers/gmlp/index.html)

#### ✨ [Recurrent Highway Networks](https://nn.labml.ai/recurrent_highway_networks/index.html)

#### ✨ [LSTM](https://nn.labml.ai/lstm/index.html)

#### ✨ [HyperNetworks - HyperLSTM](https://nn.labml.ai/hypernetworks/hyper_lstm.html)

#### ✨ [Capsule Networks](https://nn.labml.ai/capsule_networks/index.html)

#### ✨ [Generative Adversarial Networks](https://nn.labml.ai/gan/index.html)
* [Original GAN](https://nn.labml.ai/gan/original/index.html)
* [GAN with deep convolutional network](https://nn.labml.ai/gan/dcgan/index.html)
* [Cycle GAN](https://nn.labml.ai/gan/cycle_gan/index.html)
* [Wasserstein GAN](https://nn.labml.ai/gan/wasserstein/index.html)
* [Wasserstein GAN with Gradient Penalty](https://nn.labml.ai/gan/wasserstein/gradient_penalty/index.html)
* [Style GAN 2](https://nn.labml.ai/gan/stylegan/index.html)

#### ✨ [Sketch RNN](https://nn.labml.ai/sketch_rnn/index.html)

#### ✨ [Reinforcement Learning](https://nn.labml.ai/rl/index.html)
* [Proximal Policy Optimization](https://nn.labml.ai/rl/ppo/index.html) with
 [Generalized Advantage Estimation](https://nn.labml.ai/rl/ppo/gae.html)
* [Deep Q Networks](https://nn.labml.ai/rl/dqn/index.html) with
 with [Dueling Network](https://nn.labml.ai/rl/dqn/model.html),
 [Prioritized Replay](https://nn.labml.ai/rl/dqn/replay_buffer.html)
 and Double Q Network.

#### ✨ [Optimizers](https://nn.labml.ai/optimizers/index.html)
* [Adam](https://nn.labml.ai/optimizers/adam.html)
* [AMSGrad](https://nn.labml.ai/optimizers/amsgrad.html)
* [Adam Optimizer with warmup](https://nn.labml.ai/optimizers/adam_warmup.html)
* [Noam Optimizer](https://nn.labml.ai/optimizers/noam.html)
* [Rectified Adam Optimizer](https://nn.labml.ai/optimizers/radam.html)
* [AdaBelief Optimizer](https://nn.labml.ai/optimizers/ada_belief.html)

#### ✨ [Normalization Layers](https://nn.labml.ai/normalization/index.html)
* [Batch Normalization](https://nn.labml.ai/normalization/batch_norm/index.html)
* [Layer Normalization](https://nn.labml.ai/normalization/layer_norm/index.html)
* [Instance Normalization](https://nn.labml.ai/normalization/instance_norm/index.html)
* [Group Normalization](https://nn.labml.ai/normalization/group_norm/index.html)
* [Weight Standardization](https://nn.labml.ai/normalization/weight_standardization/index.html)
* [Batch-Channel Normalization](https://nn.labml.ai/normalization/batch_channel_norm/index.html)

### Installation

```bash
pip install labml-nn
```

### Citing LabML

If you use LabML for academic research, please cite the library using the following BibTeX entry.

```bibtex
@misc{labml,
 author = {Varuna Jayasiri, Nipun Wijerathne},
 title = {LabML: A library to organize machine learning experiments},
 year = {2020},
 url = {https://nn.labml.ai/},
}
```
