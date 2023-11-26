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
# from labml_nn.diffusion.stable_diffusion.model.autoencoder import Encoder, Decoder, Autoencoder
from labml_nn.diffusion.stable_diffusion.model.clip_embedder import CLIPTextEmbedder, CLIPImageEmbedder
from libs.caption_decoder import CaptionDecoder
import libs.autoencoder
# from consistencydecoder import ConsistencyDecoder


def set_seed(seed: int):
    """
    ### Set random seeds
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def load_model(path: Path = None, config=None) -> LatentDiffusion:
    """
    ### Load [`LatentDiffusion` model](latent_diffusion.html)
    """

    if config.localtest == 1:
        version = "openai/clip-vit-large-patch14"
        model = "ViT-B/32"
        download_root = "home/wuyujia/.cache/clip/decoder.pt"
    else:
        version = "/other_models/models--openai--clip-vit-large-patch14/snapshots/8d052a0f05efbaefbc9e8786ba291cfdf93e5bff"
        model = "/other_models/clip/ViT-B-32.pt"
        download_root = "/other_models/clip/decoder.pt"
        
    with monit.section('Initialize autoencoder'):
        autoencoder = libs.autoencoder.get_model(**config.autoencoder).to(config.device)
        autoencoder.requires_grad = False
        
    # with monit.section('Initialize ConsistencyDecoder'), torch.cuda.amp.autocast(), torch.no_grad():
    #     decoder_consistency = ConsistencyDecoder(device=config.device, download_root=download_root)
        
    # Initialize the CLIP text embedder
    with monit.section('Initialize CLIP Embedder'):
        clip_text_embedder = CLIPTextEmbedder(version,device=config.device)
    # Initialize the ClIP image embber
    with monit.section('Initialize CLIP Image Embedder'):
        clip_img_embedder = CLIPImageEmbedder(model,device=config.device)
    with monit.section('Initialize CaptionDecoder'):
        caption_decoder = CaptionDecoder(**config.caption_decoder, device=config.device)
        
    # Initialize the U-ViT 
    with monit.section('Initialize U-ViT'):
        from libs.uvit_multi_post_ln_v1 import UViT
        nnet_model = UViT(**config.nnet)

    # Initialize the Latent Diffusion model
    with monit.section('Initialize Latent Diffusion model'):
        model = LatentDiffusion(linear_start=0.00085,
                                linear_end=0.0120,
                                n_steps=1000,
                                latent_scaling_factor=0.18215,
                                # decoder_consistency=decoder_consistency,
                                autoencoder=autoencoder,
                                clip_embedder=clip_text_embedder,
                                clip_img_embedder=clip_img_embedder,
                                caption_decoder=caption_decoder,
                                nnet_model=nnet_model)

    # Load the checkpoint
    with monit.section(f"Loading model from {path}"):
        checkpoint = torch.load(path, map_location="cpu")

    # Set model state
    with monit.section('Load state'):
        missing_keys, extra_keys = nnet_model.load_state_dict(checkpoint, False)

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

    image = image.resize((512, 512))
    # # Convert to numpy and map to `[-1, 1]` for `[0, 255]`
    # image = np.array(image).to(float32)* (2. / 255.0) - 1
    # # Transpose to shape `[batch_size, channels, height, width]`
    # image = image[None].transpose(0, 3, 1, 2)

    
    image_np = np.array(image).transpose(2, 0, 1)
                
    # Convert to tensor and normalize
    image_tensor = torch.tensor(image_np, dtype=torch.float32)
    image = 2 * (image_tensor / 255) - 1
    image = image.unsqueeze(0)

def load_img_rm(image):

    image = image.resize((512, 512))
    # # Convert to numpy and map to `[-1, 1]` for `[0, 255]`
    # image = np.array(image).to(float32)* (2. / 255.0) - 1
    # # Transpose to shape `[batch_size, channels, height, width]`
    # image = image[None].transpose(0, 3, 1, 2)

    
    image_np = np.array(image).transpose(2, 0, 1)
                
    # Convert to tensor and normalize
    image_tensor = torch.tensor(image_np, dtype=torch.float32)
    image = 2 * (image_tensor / 255) - 1
    image = image.unsqueeze(0)


    # Convert to torch
    return image


def save_images(images: torch.Tensor, dest_path: str, prefix: str = '', img_format: str = 'jpeg'):
    """
    ### Save a images

    :param images: is the tensor with images of shape `[batch_size, channels, height, width]`
    :param dest_path: is the folder to save images in
    :param prefix: is the prefix to add to file names
    :param img_format: is the image format
    
    
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)
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
       
        # img = Image.fromarray((255. * img).astype(np.float32))
        img.save(os.path.join(dest_path, f"{prefix}{i:05}.{img_format}"), format=img_format)
        # img.save(os.path.join(dest_path, f"{prefix}{i:05}.{img_format}"), format=img_format)

# 自己加的处理批量图片
def load_imgs(paths):
    """
    ### Load multiple images

    This loads multiple images from files and returns a list of PyTorch tensors.

    :param paths: is a list of image paths
    """
    # Create an empty list to store the images
    images = []
    # Loop over the paths
    for path in paths:
        # Load the image using the `load_img` function
        image = load_imgs(path)
        # Append the image to the list
        images.append(image)
    # Return the list of images
    return images
