"""
---
title: Generate images using stable diffusion with a prompt
summary: >
 Generate images using stable diffusion with a prompt
---

# Generate images using [stable diffusion](../index.html) with a prompt
"""

import argparse
import os
from pathlib import Path

import torch

from labml import lab, monit
from labml_nn.diffusion.stable_diffusion.latent_diffusion import LatentDiffusion
from labml_nn.diffusion.stable_diffusion.sampler.ddim import DDIMSampler
from labml_nn.diffusion.stable_diffusion.sampler.ddpm import DDPMSampler
from labml_nn.diffusion.stable_diffusion.util import load_model, save_images, set_seed


class Txt2Img:
    """
    ### Text to image class
    """
    model: LatentDiffusion

    def __init__(self, *,
                 checkpoint_path: Path,
                 sampler_name: str,
                 n_steps: int = 50,
                 ddim_eta: float = 0.0,
                 ):
        """
        :param checkpoint_path: is the path of the checkpoint
        :param sampler_name: is the name of the [sampler](../sampler/index.html)
        :param n_steps: is the number of sampling steps
        :param ddim_eta: is the [DDIM sampling](../sampler/ddim.html) $\eta$ constant
        """
        # Load [latent diffusion model](../latent_diffusion.html)
        self.model = load_model(checkpoint_path)
        # Get device
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        # Move the model to device
        self.model.to(self.device)

        # Initialize [sampler](../sampler/index.html)
        if sampler_name == 'ddim':
            self.sampler = DDIMSampler(self.model,
                                       n_steps=n_steps,
                                       ddim_eta=ddim_eta)
        elif sampler_name == 'ddpm':
            self.sampler = DDPMSampler(self.model)

    @torch.no_grad()
    def __call__(self, *,
                 dest_path: str,
                 batch_size: int = 3,
                 prompt: str,
                 h: int = 512, w: int = 512,
                 uncond_scale: float = 7.5,
                 ):
        """
        :param dest_path: is the path to store the generated images
        :param batch_size: is the number of images to generate in a batch
        :param prompt: is the prompt to generate images with
        :param h: is the height of the image
        :param w: is the width of the image
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        """
        # Number of channels in the image
        c = 4
        # Image to latent space resolution reduction
        f = 8

        # Make a batch of prompts
        prompts = batch_size * [prompt]

        # AMP auto casting
        with torch.cuda.amp.autocast():
            # In unconditional scaling is not $1$ get the embeddings for empty prompts (no conditioning).
            if uncond_scale != 1.0:
                un_cond = self.model.get_text_conditioning(batch_size * [""])
            else:
                un_cond = None
            # Get the prompt embeddings
            cond = self.model.get_text_conditioning(prompts)
            # [Sample in the latent space](../sampler/index.html).
            # `x` will be of shape `[batch_size, c, h / f, w / f]`
            x = self.sampler.sample(cond=cond,
                                    shape=[batch_size, c, h // f, w // f],
                                    uncond_scale=uncond_scale,
                                    uncond_cond=un_cond)
            # Decode the image from the [autoencoder](../model/autoencoder.html)
            images = self.model.autoencoder_decode(x)

        # Save images
        save_images(images, dest_path, 'txt_')


def main():
    """
    ### CLI
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument("--batch_size", type=int, default=4, help="batch size")

    parser.add_argument(
        '--sampler',
        dest='sampler_name',
        choices=['ddim', 'ddpm'],
        default='ddim',
        help=f'Set the sampler.',
    )

    parser.add_argument("--flash", action='store_true', help="whether to use flash attention")

    parser.add_argument("--steps", type=int, default=50, help="number of sampling steps")

    parser.add_argument("--scale", type=float, default=7.5,
                        help="unconditional guidance scale: "
                             "eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))")

    opt = parser.parse_args()

    set_seed(42)

    # Set flash attention
    from labml_nn.diffusion.stable_diffusion.model.unet_attention import CrossAttention
    CrossAttention.use_flash_attention = opt.flash

    #
    txt2img = Txt2Img(checkpoint_path=lab.get_data_path() / 'stable-diffusion' / 'sd-v1-4.ckpt',
                      sampler_name=opt.sampler_name,
                      n_steps=opt.steps)

    with monit.section('Generate'):
        txt2img(dest_path='outputs',
                batch_size=opt.batch_size,
                prompt=opt.prompt,
                uncond_scale=opt.scale)


#
if __name__ == "__main__":
    main()
