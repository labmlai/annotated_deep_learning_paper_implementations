"""
# [LabML Neural Networks](index.html)

This is a collection of simple PyTorch implementations of
neural networks and related algorithms.
[These implementations](https://github.com/lab-ml/nn) are documented with explanations,
and the [website](index.html)
renders these as side-by-side formatted notes.
We believe these would help you understand these algorithms better.

We are actively maintaining this repo and adding new
implementations.

## Modules

#### ✨ [Transformers](transformers/index.html)

* [Multi-headed attention](transformers/mha.html)
* [Transformer building blocks](transformers/models.html)
* [Transformer XL](transformers/xl/index.html)
    * [Relative multi-headed attention](transformers/xl/relative_mha.html)
* [GPT Architecture](transformers/gpt/index.html)
* [GLU Variants](transformers/glu_variants/simple.html)
* [kNN-LM: Generalization through Memorization](transformers/knn/index.html)
* [Feedback Transformer](transformers/feedback/index.html)
* [Switch Transformer](transformers/switch/index.html)

#### ✨ [Recurrent Highway Networks](recurrent_highway_networks/index.html)

#### ✨ [LSTM](lstm/index.html)

#### ✨ [HyperNetworks - HyperLSTM](hypernetworks/hyper_lstm.html)

#### ✨ [Capsule Networks](capsule_networks/index.html)

#### ✨ [Generative Adversarial Networks](gan/index.html)
* [GAN with a multi-layer perceptron](gan/simple_mnist_experiment.html)
* [GAN with deep convolutional network](gan/dcgan.html)
* [Cycle GAN](gan/cycle_gan.html)

#### ✨ [Sketch RNN](sketch_rnn/index.html)

#### ✨ [Reinforcement Learning](rl/index.html)
* [Proximal Policy Optimization](rl/ppo/index.html) with
 [Generalized Advantage Estimation](rl/ppo/gae.html)
* [Deep Q Networks](rl/dqn/index.html) with
 with [Dueling Network](rl/dqn/model.html),
 [Prioritized Replay](rl/dqn/replay_buffer.html)
 and Double Q Network.

#### ✨ [Optimizers](optimizers/index.html)
* [Adam](optimizers/adam.html)
* [AMSGrad](optimizers/amsgrad.html)
* [Adam Optimizer with warmup](optimizers/adam_warmup.html)
* [Noam Optimizer](optimizers/noam.html)
* [Rectified Adam Optimizer](optimizers/radam.html)
* [AdaBelief Optimizer](optimizers/ada_belief.html)

#### ✨ [Normalization Layers](https://nn.labml.ai/normalization/index.html)
* [Batch Normalization](https://nn.labml.ai/normalization/batch_norm/index.html)
* [Layer Normalization](https://nn.labml.ai/normalization/layer_norm/index.html)

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
 url = {https://lab-ml.com/},
}
```
"""
