"""
---
title: Sampling algorithms for stable diffusion
summary: >
 Annotated PyTorch implementation/tutorial of
 sampling algorithms
 for stable diffusion model.
---

# Sampling algorithms for [stable diffusion](../index.html)

We have implemented the following [sampling algorithms](sampler/index.html):

* [Denoising Diffusion Probabilistic Models (DDPM) Sampling](ddpm.html)
* [Denoising Diffusion Implicit Models (DDIM) Sampling](ddim.html)
"""

from typing import Optional, List

import torch

from labml_nn.diffusion.stable_diffusion.latent_diffusion import LatentDiffusion


class DiffusionSampler:
    """
    ## Base class for sampling algorithms
    """
    model: LatentDiffusion

    def __init__(self, model: LatentDiffusion):
        """
        :param model: is the model to predict noise $\epsilon_\text{cond}(x_t, c)$
        """
        super().__init__()
        # Set the model $\epsilon_\text{cond}(x_t, c)$
        self.model = model
        # Get number of steps the model was trained with $T$
        self.n_steps = model.n_steps

    def get_eps(self,
                x: torch.Tensor, 
                t: torch.Tensor,
                c: torch.Tensor,
                clip_img: torch.Tensor,
                t_text: torch.Tensor,
                datatype: torch.Tensor,
                captiondecodeprefix, captionencodeprefix, *,
                uncond_scale: float, uncond_cond: Optional[torch.Tensor]):
        """
        ## Get $\epsilon(x_t, c)$

        :param x: is $x_t$ of shape `[batch_size, channels, height, width]`
        :param t: is $t$ of shape `[batch_size]`
        :param c: is the conditional embeddings $c$ of shape `[batch_size, emb_size]`
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        """
        # When the scale $s = 1$
        # $$\epsilon_\theta(x_t, c) = \epsilon_\text{cond}(x_t, c)$$
        
        # print("Type of 'x':\n", type(x))
        # print("Shape of 'x':\n", x.shape)


        # print("Type of 't':\n", type(t))
        # print("Shape of 't':\n", t.shape)


        # print("Type of 'c':\n", type(c))
        # print("Shape of 'c':\n", c.shape)


        # print("Type of 'clip_img':\n", type(clip_img))
        # print("Shape of 'clip_img':\n", clip_img.shape)


        # print("Type of 't_text':\n", type(t_text))
        # print("Shape of 't_text':\n", t_text.shape)


        # print("Type of 'datatype':\n", type(datatype))
        # print("Shape of 'datatype':\n", datatype.shape)


        # print("Type of 'captiondecodeprefix':\n", type(captiondecodeprefix))

        # print("Type of 'captionencodeprefix':\n", type(captionencodeprefix))


        # print("Type of 'uncond_scale':\n", type(uncond_scale))

        # if uncond_cond is not None:
        #     print("Printing variable 'uncond_cond':\n", uncond_cond)
        #     print("Type of 'uncond_cond':\n", type(uncond_cond))
        #     print("Shape of 'uncond_cond':\n", uncond_cond.shape)
        # else:
        #     print("uncond_cond is None")
        
        
        
     
        if uncond_cond is None or uncond_scale == 1.:
            # print("uncond_scale is None or ==1")
            return self.model(x,clip_img, c, t, t_text, datatype)


        # Duplicate $x_t$ and $t$
        x_in = torch.cat([x] * 2)
        ### x_in: torch.Size([8, 4, 80, 60])

        
        clip_img_in = torch.cat([clip_img] * 2)
        ###  clip_img_in: torch.Size([8, 1, 512]) 
        
        
        # clip_img_in = torch.cat([clip_img, clip_img], dim=1)

        ### t : tensor([721, 721, 721, 721], device='cuda:7')
        t_in = torch.cat([t] * 2)
        ### t_in: tensor([721, 721, 721, 721, 721, 721, 721, 721], device='cuda:7') shape = 8
        
        ### t_text: tensor([0, 0, 0, 0], device='cuda:7', dtype=torch.int32)
        t_text_in = torch.cat([t_text] * 2)
        ### t_text_in: tensor([0, 0, 0, 0, 0, 0, 0, 0], device='cuda:7', dtype=torch.int32)
        
        # Concatenated $c$ and $c_u$
        ## 这里应该是 768，在 decoder 之前是 64
        
    ### c: torch.Size([4, 77, 64])  c_in: torch.Size([4, 77, 768])
        c = captiondecodeprefix(c)
        ### c.shape : torch.Size([4, 77, 768])
 
        ### uncond_cond.shape: torch.Size([4, 77, 768])
        # c = c.view(1,77,1536)
        # print(c.shape)
        # print(uncond_cond.shape) 
        c_in = torch.cat([uncond_cond, c])
        
        ### c_in.shape: torch.Size([8, 77, 768])

        # Get $\epsilon_\text{cond}(x_t, c)$ and $\epsilon_\text{cond}(x_t, c_u)$
        
        c_in = captionencodeprefix(c_in)
        ### c_in.shape: torch.Size([8, 77, 64])
        
        datatype_in = torch.cat([datatype] * 2)
        # inchanel_in = 1

        result,_,_ = self.model(x_in, clip_img_in, c_in, t_in, t_text_in, datatype_in)
        # print(result)
       
        e_t_uncond, e_t_cond = torch.chunk(result,2,dim=0)
        
      
        ### e_t_uncond.shape: torch.Size([8, 4, 64, 64])
        ### e_t_cond.shape : torch.Size([8, 1, 512])
        # Calculate
        # $$\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$$
        e_t = e_t_uncond + uncond_scale * (e_t_cond - e_t_uncond)
        return e_t

    def sample(self,
               shape: List[int],
               clip_img: torch.Tensor,
               cond: torch.Tensor,
               repeat_noise: bool = False,
               temperature: float = 1.,
               x_last: Optional[torch.Tensor] = None,
               uncond_scale: float = 1.,
               uncond_cond: Optional[torch.Tensor] = None,
               skip_steps: int = 0,
               ):
        """
        ### Sampling Loop

        :param shape: is the shape of the generated images in the
            form `[batch_size, channels, height, width]`
        :param cond: is the conditional embeddings $c$
        :param temperature: is the noise temperature (random noise gets multiplied by this)
        :param x_last: is $x_T$. If not provided random noise will be used.
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        :param skip_steps: is the number of time steps to skip.
        """
        raise NotImplementedError()

    def paint(self, x: torch.Tensor, cond: torch.Tensor, t_start: int, *,
              orig: Optional[torch.Tensor] = None,
              mask: Optional[torch.Tensor] = None, orig_noise: Optional[torch.Tensor] = None,
              uncond_scale: float = 1.,
              uncond_cond: Optional[torch.Tensor] = None,
              ):
        """
        ### Painting Loop

        :param x: is $x_{T'}$ of shape `[batch_size, channels, height, width]`
        :param cond: is the conditional embeddings $c$
        :param t_start: is the sampling step to start from, $T'$
        :param orig: is the original image in latent page which we are in paining.
        :param mask: is the mask to keep the original image.
        :param orig_noise: is fixed noise to be added to the original image.
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        """
        raise NotImplementedError()

    def q_sample(self, x0: torch.Tensor, index: int, noise: Optional[torch.Tensor] = None):
        """
        ### Sample from $q(x_t|x_0)$

        :param x0: is $x_0$ of shape `[batch_size, channels, height, width]`
        :param index: is the time step $t$ index
        :param noise: is the noise, $\epsilon$
        """
        raise NotImplementedError()



