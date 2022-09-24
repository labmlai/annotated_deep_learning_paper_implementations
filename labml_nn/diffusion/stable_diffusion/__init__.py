"""
---
title: Stable Diffusion
summary: >
 Annotated PyTorch implementation/tutorial of stable diffusion.
---

# Stable Diffusion

This is based on official stable diffusion repository
 [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion).
We have kept the model structure same so that open sourced weights could be directly loaded.
Our implementation does not contain training code.

### [PromptArt](https://promptart.labml.ai)

![PromptArt](https://labml.ai/images/promptart-feed.webp)

We have deployed a stable diffusion based image generation service
at [promptart.labml.ai](https://promptart.labml.ai)

### [Latent Diffusion Model](latent_diffusion.html)

The core is the [Latent Diffusion Model](latent_diffusion.html).
It consists of:

* [AutoEncoder](model/autoencoder.html)
* [U-Net](model/unet.html) with [attention](model/unet_attention.html)

We have also (optionally) integrated [Flash Attention](https://github.com/HazyResearch/flash-attention)
into our [U-Net attention](model/unet_attention.html) which lets you speed up
the performance by close to 50% on an RTX A6000 GPU.

The diffusion is conditioned based on [CLIP embeddings](model/clip_embedder.html).

### [Sampling Algorithms](sampler/index.html)

We have implemented the following [sampling algorithms](sampler/index.html):

* [Denoising Diffusion Probabilistic Models (DDPM) Sampling](sampler/ddpm.html)
* [Denoising Diffusion Implicit Models (DDIM) Sampling](sampler/ddim.html)

### [Example Scripts](scripts/index.html)

Here are the image generation scripts:

* [Generate images from text prompts](scripts/text_to_image.html)
* [Generate images based on a given image, guided by a prompt](scripts/image_to_image.html)
* [Modify parts of a given image based on a text prompt](scripts/in_paint.html)

#### [Utilities](util.html)

[`util.py`](util.html) defines the utility functions.
"""
