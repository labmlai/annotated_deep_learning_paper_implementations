"""
# [labml.ai Annotated PyTorch Paper Implementations](index.html)

This is a collection of simple PyTorch implementations of
neural networks and related algorithms.
[These implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations) are documented with explanations,
and the [website](index.html)
renders these as side-by-side formatted notes.
We believe these would help you understand these algorithms better.

![Screenshot](dqn-light.png)

We are actively maintaining this repo and adding new
implementations.
[![Twitter](https://img.shields.io/twitter/follow/labmlai?style=social)](https://twitter.com/labmlai) for updates.

## Paper Implementations

#### ✨ [Transformers](transformers/index.html)

* [Multi-headed attention](transformers/mha.html)
* [Transformer building blocks](transformers/models.html)
* [Transformer XL](transformers/xl/index.html)
    * [Relative multi-headed attention](transformers/xl/relative_mha.html)
* [Rotary Positional Embeddings (RoPE)](transformers/rope/index.html)
* [Attention with Linear Biases (ALiBi)](transformers/alibi/index.html)
* [RETRO](transformers/retro/index.html)
* [Compressive Transformer](transformers/compressive/index.html)
* [GPT Architecture](transformers/gpt/index.html)
* [GLU Variants](transformers/glu_variants/simple.html)
* [kNN-LM: Generalization through Memorization](transformers/knn/index.html)
* [Feedback Transformer](transformers/feedback/index.html)
* [Switch Transformer](transformers/switch/index.html)
* [Fast Weights Transformer](transformers/fast_weights/index.html)
* [FNet](transformers/fnet/index.html)
* [Attention Free Transformer](transformers/aft/index.html)
* [Masked Language Model](transformers/mlm/index.html)
* [MLP-Mixer: An all-MLP Architecture for Vision](transformers/mlp_mixer/index.html)
* [Pay Attention to MLPs (gMLP)](transformers/gmlp/index.html)
* [Vision Transformer (ViT)](transformers/vit/index.html)
* [Primer EZ](transformers/primer_ez/index.html)
* [Hourglass](transformers/hour_glass/index.html)

#### ✨ [Eleuther GPT-NeoX](neox/index.html)
* [Generate on a 48GB GPU](neox/samples/generate.html)
* [Finetune on two 48GB GPUs](neox/samples/finetune.html)
* [LLM.int8()](neox/utils/llm_int8.html)

#### ✨ [Diffusion models](diffusion/index.html)

* [Denoising Diffusion Probabilistic Models (DDPM)](diffusion/ddpm/index.html)
* [Denoising Diffusion Implicit Models (DDIM)](diffusion/stable_diffusion/sampler/ddim.html)
* [Latent Diffusion Models](diffusion/stable_diffusion/latent_diffusion.html)
* [Stable Diffusion](diffusion/stable_diffusion/index.html)

#### ✨ [Generative Adversarial Networks](gan/index.html)
* [Original GAN](gan/original/index.html)
* [GAN with deep convolutional network](gan/dcgan/index.html)
* [Cycle GAN](gan/cycle_gan/index.html)
* [Wasserstein GAN](gan/wasserstein/index.html)
* [Wasserstein GAN with Gradient Penalty](gan/wasserstein/gradient_penalty/index.html)
* [StyleGAN 2](gan/stylegan/index.html)

#### ✨ [Recurrent Highway Networks](recurrent_highway_networks/index.html)

#### ✨ [LSTM](lstm/index.html)

#### ✨ [HyperNetworks - HyperLSTM](hypernetworks/hyper_lstm.html)

#### ✨ [ResNet](resnet/index.html)

#### ✨ [ConvMixer](conv_mixer/index.html)

#### ✨ [Capsule Networks](capsule_networks/index.html)

#### ✨ [U-Net](unet/index.html)

#### ✨ [Sketch RNN](sketch_rnn/index.html)

#### ✨ Graph Neural Networks

* [Graph Attention Networks (GAT)](graphs/gat/index.html)
* [Graph Attention Networks v2 (GATv2)](graphs/gatv2/index.html)

#### ✨ [Reinforcement Learning](rl/index.html)
* [Proximal Policy Optimization](rl/ppo/index.html) with
 [Generalized Advantage Estimation](rl/ppo/gae.html)
* [Deep Q Networks](rl/dqn/index.html) with
 with [Dueling Network](rl/dqn/model.html),
 [Prioritized Replay](rl/dqn/replay_buffer.html)
 and Double Q Network.

#### ✨ [Counterfactual Regret Minimization (CFR)](cfr/index.html)

Solving games with incomplete information such as poker with CFR.

* [Kuhn Poker](cfr/kuhn/index.html)

#### ✨ [Optimizers](optimizers/index.html)
* [Adam](optimizers/adam.html)
* [AMSGrad](optimizers/amsgrad.html)
* [Adam Optimizer with warmup](optimizers/adam_warmup.html)
* [Noam Optimizer](optimizers/noam.html)
* [Rectified Adam Optimizer](optimizers/radam.html)
* [AdaBelief Optimizer](optimizers/ada_belief.html)

#### ✨ [Normalization Layers](normalization/index.html)
* [Batch Normalization](normalization/batch_norm/index.html)
* [Layer Normalization](normalization/layer_norm/index.html)
* [Instance Normalization](normalization/instance_norm/index.html)
* [Group Normalization](normalization/group_norm/index.html)
* [Weight Standardization](normalization/weight_standardization/index.html)
* [Batch-Channel Normalization](normalization/batch_channel_norm/index.html)
* [DeepNorm](normalization/deep_norm/index.html)

#### ✨ [Distillation](distillation/index.html)

#### ✨ [Adaptive Computation](adaptive_computation/index.html)

* [PonderNet](adaptive_computation/ponder_net/index.html)

#### ✨ [Uncertainty](uncertainty/index.html)

* [Evidential Deep Learning to Quantify Classification Uncertainty](uncertainty/evidence/index.html)

#### ✨ [Activations](activations/index.html)

* [Fuzzy Tiling Activations](activations/fta/index.html)

#### ✨ [Language Model Sampling Techniques](sampling/index.html)
* [Greedy Sampling](sampling/greedy.html)
* [Temperature Sampling](sampling/temperature.html)
* [Top-k Sampling](sampling/top_k.html)
* [Nucleus Sampling](sampling/nucleus.html)

#### ✨ [Scalable Training/Inference](scaling/index.html)
* [Zero3 memory optimizations](scaling/zero3/index.html)

## Highlighted Research Paper PDFs

* [Autoregressive Search Engines: Generating Substrings as Document Identifiers](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/papers/2204.10628.pdf)
* [Training Compute-Optimal Large Language Models](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/papers/2203.15556.pdf)
* [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/papers/1910.02054.pdf)
* [PaLM: Scaling Language Modeling with Pathways](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/papers/2204.02311.pdf)
* [Hierarchical Text-Conditional Image Generation with CLIP Latents](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/papers/dall-e-2.pdf)
* [STaR: Self-Taught Reasoner Bootstrapping Reasoning With Reasoning](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/papers/2203.14465.pdf)
* [Improving language models by retrieving from trillions of tokens](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/papers/2112.04426.pdf)
* [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/papers/2003.08934.pdf)
* [Attention Is All You Need](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/papers/1706.03762.pdf)
* [Denoising Diffusion Probabilistic Models](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/papers/2006.11239.pdf)
* [Primer: Searching for Efficient Transformers for Language Modeling](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/papers/2109.08668.pdf)
* [On First-Order Meta-Learning Algorithms](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/papers/1803.02999.pdf)
* [Learning Transferable Visual Models From Natural Language Supervision](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/papers/2103.00020.pdf)
* [The Sensory Neuron as a Transformer: Permutation-Invariant Neural Networks for Reinforcement Learning](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/papers/2109.02869.pdf)
* [Meta-Gradient Reinforcement Learning](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/papers/1805.09801.pdf)
* [ETA Prediction with Graph Neural Networks in Google Maps](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/papers/google_maps_eta.pdf)
* [PonderNet: Learning to Ponder](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/papers/ponder_net.pdf)
* [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/papers/muzero.pdf)
* [GANs N’ Roses: Stable, Controllable, Diverse Image to Image Translation (works for videos too!)](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/papers/gans_n_roses.pdf)
* [An Image is Worth 16X16 Word: Transformers for Image Recognition at Scale](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/papers/vit.pdf)
* [Deep Residual Learning for Image Recognition](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/papers/resnet.pdf)
* [Distilling the Knowledge in a Neural Network](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/papers/distillation.pdf)

### Installation

```bash
pip install labml-nn
```

### Citing LabML

If you use this for academic research, please cite it using the following BibTeX entry.

```bibtex
@misc{labml,
 author = {Varuna Jayasiri, Nipun Wijerathne},
 title = {labml.ai Annotated Paper Implementations},
 year = {2020},
 url = {},
}
```
"""
