# [Denoising Diffusion Probabilistic Models (DDPM)](https://nn.labml.ai/diffusion/ddpm/index.html)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/experiment.ipynb)
[![Open In Comet](https://images.labml.ai/images/comet.svg?experiment=capsule_networks&file=model)](https://www.comet.com/labml/diffuse/view/FknjSiKWotr8fgZerpC1sV1cy/panels?utm_source=referral&utm_medium=partner&utm_campaign=labml)

This is a [PyTorch](https://pytorch.org) implementation/tutorial of the paper
[Denoising Diffusion Probabilistic Models](https://papers.labml.ai/paper/2006.11239).

In simple terms, we get an image from data and add noise step by step.
Then We train a model to predict that noise at each step and use the model to
generate images.

Here is the [UNet model](https://nn.labml.ai/diffusion/ddpm/unet.html) that predicts the noise and
[training code](https://nn.labml.ai/diffusion/ddpm/experiment.html).
[This file](https://nn.labml.ai/diffusion/ddpm/evaluate.html) can generate samples and interpolations
from a trained model.
