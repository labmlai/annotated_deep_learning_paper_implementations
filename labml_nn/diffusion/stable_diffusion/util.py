"""
---
title: Utility functions for stable diffusion
summary: >
 Utility functions for stable diffusion
---

# Utility functions for [stable diffusion](index.html)
"""

import os
import random
from pathlib import Path

import PIL
import numpy as np
import torch
from PIL import Image

from labml import monit
from labml.logger import inspect
from labml_nn.diffusion.stable_diffusion.latent_diffusion import LatentDiffusion
from labml_nn.diffusion.stable_diffusion.model.autoencoder import Encoder, Decoder, Autoencoder
from labml_nn.diffusion.stable_diffusion.model.clip_embedder import CLIPTextEmbedder
from labml_nn.diffusion.stable_diffusion.model.unet import UNetModel


def set_seed(seed: int):
    """
    ### Set random seeds
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(path: Path = None) -> LatentDiffusion:
    """
    ### Load [`LatentDiffusion` model](latent_diffusion.html)
    """

    # Initialize the autoencoder
    with monit.section('Initialize autoencoder'):
        encoder = Encoder(z_channels=4,
                          in_channels=3,
                          channels=128,
                          channel_multipliers=[1, 2, 4, 4],
                          n_resnet_blocks=2)

        decoder = Decoder(out_channels=3,
                          z_channels=4,
                          channels=128,
                          channel_multipliers=[1, 2, 4, 4],
                          n_resnet_blocks=2)

        autoencoder = Autoencoder(emb_channels=4,
                                  encoder=encoder,
                                  decoder=decoder,
                                  z_channels=4)

    # Initialize the CLIP text embedder
    with monit.section('Initialize CLIP Embedder'):
        clip_text_embedder = CLIPTextEmbedder()

    # Initialize the U-Net
    with monit.section('Initialize U-Net'):
        unet_model = UNetModel(in_channels=4,
                               out_channels=4,
                               channels=320,
                               attention_levels=[0, 1, 2],
                               n_res_blocks=2,
                               channel_multipliers=[1, 2, 4, 4],
                               n_heads=8,
                               tf_layers=1,
                               d_cond=768)

    # Initialize the Latent Diffusion model
    with monit.section('Initialize Latent Diffusion model'):
        model = LatentDiffusion(linear_start=0.00085,
                                linear_end=0.0120,
                                n_steps=1000,
                                latent_scaling_factor=0.18215,

                                autoencoder=autoencoder,
                                clip_embedder=clip_text_embedder,
                                unet_model=unet_model)

    # Load the checkpoint
    with monit.section(f"Loading model from {path}"):
        checkpoint = torch.load(path, map_location="cpu")

    # Set model state
    with monit.section('Load state'):
        missing_keys, extra_keys = model.load_state_dict(checkpoint["state_dict"], strict=False)

    # Debugging output
    inspect(global_step=checkpoint.get('global_step', -1), missing_keys=missing_keys, extra_keys=extra_keys,
            _expand=True)

    #
    model.eval()
    return model


def load_img(path: str):
    """
    ### Load an image

    This loads an image from a file and returns a PyTorch tensor.

    :param path: is the path of the image
    """
    # Open Image
    image = Image.open(path).convert("RGB")
    # Get image size
    w, h = image.size
    # Resize to a multiple of 32
    w = w - w % 32
    h = h - h % 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    # Convert to numpy and map to `[-1, 1]` for `[0, 255]`
    image = np.array(image).astype(np.float32) * (2. / 255.0) - 1
    # Transpose to shape `[batch_size, channels, height, width]`
    image = image[None].transpose(0, 3, 1, 2)
    # Convert to torch
    return torch.from_numpy(image)


def save_images(images: torch.Tensor, dest_path: str, prefix: str = '', img_format: str = 'jpeg'):
    """
    ### Save a images

    :param images: is the tensor with images of shape `[batch_size, channels, height, width]`
    :param dest_path: is the folder to save images in
    :param prefix: is the prefix to add to file names
    :param img_format: is the image format
    """

    # Create the destination folder
    os.makedirs(dest_path, exist_ok=True)

    # Map images to `[0, 1]` space and clip
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    # Transpose to `[batch_size, height, width, channels]` and convert to numpy
    images = images.cpu().permute(0, 2, 3, 1).numpy()

    # Save images
    for i, img in enumerate(images):
        img = Image.fromarray((255. * img).astype(np.uint8))
        img.save(os.path.join(dest_path, f"{prefix}{i:05}.{img_format}"), format=img_format)
