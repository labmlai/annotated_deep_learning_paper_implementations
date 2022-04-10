[![Twitter](https://img.shields.io/twitter/follow/labmlai?style=social)](https://twitter.com/labmlai)

# [labml.ai Deep Learning Paper Implementations](https://nn.labml.ai/index.html)

This is a collection of simple PyTorch implementations of
neural networks and related algorithms.
These implementations are documented with explanations,

[The website](https://nn.labml.ai/index.html)
renders these as side-by-side formatted notes.
We believe these would help you understand these algorithms better.

![Screenshot](https://nn.labml.ai/dqn-light.png)

We are actively maintaining this repo and adding new 
implementations almost weekly.
[![Twitter](https://img.shields.io/twitter/follow/labmlai?style=social)](https://twitter.com/labmlai) for updates.

## Modules

#### ✨ [Transformers](https://nn.labml.ai/transformers/index.html)

* [Multi-headed attention](https://nn.labml.ai/transformers/mha.html)
* [Transformer building blocks](https://nn.labml.ai/transformers/models.html) 
* [Transformer XL](https://nn.labml.ai/transformers/xl/index.html)
    * [Relative multi-headed attention](https://nn.labml.ai/transformers/xl/relative_mha.html)
* [Rotary Positional Embeddings](https://nn.labml.ai/transformers/rope/index.html)
* [RETRO](https://nn.labml.ai/transformers/retro/index.html)
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
* [Vision Transformer (ViT)](https://nn.labml.ai/transformers/vit/index.html)
* [Primer EZ](https://nn.labml.ai/transformers/primer_ez/index.html)
* [Hourglass](https://nn.labml.ai/transformers/hour_glass/index.html)

#### ✨ [Recurrent Highway Networks](https://nn.labml.ai/recurrent_highway_networks/index.html)

#### ✨ [LSTM](https://nn.labml.ai/lstm/index.html)

#### ✨ [HyperNetworks - HyperLSTM](https://nn.labml.ai/hypernetworks/hyper_lstm.html)

#### ✨ [ResNet](https://nn.labml.ai/resnet/index.html)

#### ✨ [ConvMixer](https://nn.labml.ai/conv_mixer/index.html)

#### ✨ [Capsule Networks](https://nn.labml.ai/capsule_networks/index.html)

#### ✨ [Generative Adversarial Networks](https://nn.labml.ai/gan/index.html)
* [Original GAN](https://nn.labml.ai/gan/original/index.html)
* [GAN with deep convolutional network](https://nn.labml.ai/gan/dcgan/index.html)
* [Cycle GAN](https://nn.labml.ai/gan/cycle_gan/index.html)
* [Wasserstein GAN](https://nn.labml.ai/gan/wasserstein/index.html)
* [Wasserstein GAN with Gradient Penalty](https://nn.labml.ai/gan/wasserstein/gradient_penalty/index.html)
* [StyleGAN 2](https://nn.labml.ai/gan/stylegan/index.html)

#### ✨ [Diffusion models](https://nn.labml.ai/diffusion/index.html)

* [Denoising Diffusion Probabilistic Models (DDPM)](https://nn.labml.ai/diffusion/ddpm/index.html)


#### ✨ [Sketch RNN](https://nn.labml.ai/sketch_rnn/index.html)

#### ✨ Graph Neural Networks

* [Graph Attention Networks (GAT)](https://nn.labml.ai/graphs/gat/index.html)
* [Graph Attention Networks v2 (GATv2)](https://nn.labml.ai/graphs/gatv2/index.html)

#### ✨ [Counterfactual Regret Minimization (CFR)](https://nn.labml.ai/cfr/index.html)

Solving games with incomplete information such as poker with CFR.

* [Kuhn Poker](https://nn.labml.ai/cfr/kuhn/index.html)

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
* [DeepNorm](https://nn.labml.ai/normalization/deep_norm/index.html)

#### ✨ [Distillation](https://nn.labml.ai/distillation/index.html)

#### ✨ [Adaptive Computation](https://nn.labml.ai/adaptive_computation/index.html)

* [PonderNet](https://nn.labml.ai/adaptive_computation/ponder_net/index.html)

#### ✨ [Uncertainty](https://nn.labml.ai/uncertainty/index.html)

* [Evidential Deep Learning to Quantify Classification Uncertainty](https://nn.labml.ai/uncertainty/evidence/index.html)

### Installation

```bash
pip install labml-nn
```

### Citing

If you use this for academic research, please cite it using the following BibTeX entry.

```bibtex
@misc{labml,
 author = {Varuna Jayasiri, Nipun Wijerathne},
 title = {labml.ai Annotated Paper Implementations},
 year = {2020},
 url = {https://nn.labml.ai/},
}
```

### Other Projects

#### [🚀 Trending Research Papers](https://papers.labml.ai/)

This shows the most popular research papers on social media. It also aggregates links to useful resources like paper explanations videos and discussions.


#### [🧪 labml.ai/labml](https://github.com/labmlai/labml)

This is a library that let's you monitor deep learning model training and hardware usage from your mobile phone. It also comes with a bunch of other tools to help write deep learning code efficiently.

