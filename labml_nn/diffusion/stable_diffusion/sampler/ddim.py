"""
---
title: Denoising Diffusion Implicit Models (DDIM) Sampling
summary: >
 Annotated PyTorch implementation/tutorial of
 Denoising Diffusion Implicit Models (DDIM) Sampling
 for stable diffusion model.
---

# Denoising Diffusion Implicit Models (DDIM) Sampling

This implements DDIM sampling from the paper
[Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
"""

from typing import Optional, List

import numpy as np
import torch

from labml import monit
from labml_nn.diffusion.stable_diffusion.latent_diffusion import LatentDiffusion
from labml_nn.diffusion.stable_diffusion.sampler import DiffusionSampler


class DDIMSampler(DiffusionSampler):
    """
    ## DDIM Sampler

    This extends the [`DiffusionSampler` base class](index.html).

    DDIM samples images by repeatedly removing noise by sampling step by step using,

    \begin{align}
    x_{\tau_{i-1}} &= \sqrt{\alpha_{\tau_{i-1}}}\Bigg(
            \frac{x_{\tau_i} - \sqrt{1 - \alpha_{\tau_i}}\epsilon_\theta(x_{\tau_i})}{\sqrt{\alpha_{\tau_i}}}
            \Bigg) \\
            &+ \sqrt{1 - \alpha_{\tau_{i- 1}} - \sigma_{\tau_i}^2} \cdot \epsilon_\theta(x_{\tau_i}) \\
            &+ \sigma_{\tau_i} \epsilon_{\tau_i}
    \end{align}

    where $\epsilon_{\tau_i}$ is random noise,
    $\tau$ is a subsequence of $[1,2,\dots,T]$ of length $S$,
    and
    $$\sigma_{\tau_i} =
    \eta \sqrt{\frac{1 - \alpha_{\tau_{i-1}}}{1 - \alpha_{\tau_i}}}
    \sqrt{1 - \frac{\alpha_{\tau_i}}{\alpha_{\tau_{i-1}}}}$$

    Note that, $\alpha_t$ in DDIM paper refers to ${\color{lightgreen}\bar\alpha_t}$ from [DDPM](ddpm.html).
    """

    model: LatentDiffusion

    def __init__(self, model: LatentDiffusion, n_steps: int, ddim_discretize: str = "uniform", ddim_eta: float = 0.):
        """
        :param model: is the model to predict noise $\epsilon_\text{cond}(x_t, c)$
        :param n_steps: is the number of DDIM sampling steps, $S$
        :param ddim_discretize: specifies how to extract $\tau$ from $[1,2,\dots,T]$.
            It can be either `uniform` or `quad`.
        :param ddim_eta: is $\eta$ used to calculate $\sigma_{\tau_i}$. $\eta = 0$ makes the
            sampling process deterministic.
        """
        super().__init__(model)
        # Number of steps, $T$
        self.n_steps = model.n_steps

        # Calculate $\tau$ to be uniformly distributed across $[1,2,\dots,T]$
        if ddim_discretize == 'uniform':
            c = self.n_steps // n_steps
            self.time_steps = np.asarray(list(range(0, self.n_steps, c))) + 1
        # Calculate $\tau$ to be quadratically distributed across $[1,2,\dots,T]$
        elif ddim_discretize == 'quad':
            self.time_steps = ((np.linspace(0, np.sqrt(self.n_steps * .8), n_steps)) ** 2).astype(int) + 1
        else:
            raise NotImplementedError(ddim_discretize)

        with torch.no_grad():
            # Get ${\color{lightgreen}\bar\alpha_t}$
            alpha_bar = self.model.alpha_bar

            # $\alpha_{\tau_i}$
            self.ddim_alpha = alpha_bar[self.time_steps].clone().to(torch.float32)
            # $\sqrt{\alpha_{\tau_i}}$
            self.ddim_alpha_sqrt = torch.sqrt(self.ddim_alpha)
            # $\alpha_{\tau_{i-1}}$
            self.ddim_alpha_prev = torch.cat([alpha_bar[0:1], alpha_bar[self.time_steps[:-1]]])

            # $$\sigma_{\tau_i} =
            # \eta \sqrt{\frac{1 - \alpha_{\tau_{i-1}}}{1 - \alpha_{\tau_i}}}
            # \sqrt{1 - \frac{\alpha_{\tau_i}}{\alpha_{\tau_{i-1}}}}$$
            self.ddim_sigma = (ddim_eta *
                               ((1 - self.ddim_alpha_prev) / (1 - self.ddim_alpha) *
                                (1 - self.ddim_alpha / self.ddim_alpha_prev)) ** .5)

            # $\sqrt{1 - \alpha_{\tau_i}}$
            self.ddim_sqrt_one_minus_alpha = (1. - self.ddim_alpha) ** .5

    @torch.no_grad()
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
        :param x_last: is $x_{\tau_S}$. If not provided random noise will be used.
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        :param skip_steps: is the number of time steps to skip $i'$. We start sampling from $S - i'$.
            And `x_last` is then $x_{\tau_{S - i'}}$.
        """
        ### sampling loop###
        # Get device and batch size
        device = self.model.device
        bs = shape[0]

        # Get $x_{\tau_S}$
        x = x_last if x_last is not None else torch.randn(shape, device=device)

        # Time steps to sample at $\tau_{S - i'}, \tau_{S - i' - 1}, \dots, \tau_1$
        time_steps = np.flip(self.time_steps)[skip_steps:]

        for i, step in monit.enum('Sample', time_steps):
            # Index $i$ in the list $[\tau_1, \tau_2, \dots, \tau_S]$
            index = len(time_steps) - i - 1
            # Time step $\tau_i$
            ts = x.new_full((bs,), step, dtype=torch.long)

            # Sample $x_{\tau_{i-1}}$
            x, pred_x0, e_t = self.p_sample(x, clip_img, cond, step, index,
                                            repeat_noise=repeat_noise,
                                            temperature=temperature,
                                            uncond_scale=uncond_scale,
                                            uncond_cond=uncond_cond)

        # Return $x_0$
        return x

    @torch.no_grad()
    def p_sample(self, 
                 x: torch.Tensor, 
                 clip_img: torch.Tensor, 
                 c: torch.Tensor,
                 t_text: torch.Tensor,
                 t_img: torch.Tensor,
                 ts: torch.Tensor,
                 datatype: torch.Tensor,
                 captiondecodeprefix,
                 captionencodeprefix,
                 step: int,
                 orig_noise,
                 index: int,
                 *,
                 repeat_noise: bool = False,
                 temperature: float = 1.,
                 uncond_scale: float = 1.,
                 uncond_cond: Optional[torch.Tensor] = None):
        
        """
        ### Sample $x_{\tau_{i-1}}$

        :param x: is $x_{\tau_i}$ of shape `[batch_size, channels, height, width]`
        :param c: is the conditional embeddings $c$ of shape `[batch_size, emb_size]`
        :param t: is $\tau_i$ of shape `[batch_size]`
        :param step: is the step $\tau_i$ as an integer
        :param index: is index $i$ in the list $[\tau_1, \tau_2, \dots, \tau_S]$
        :param repeat_noise: specified whether the noise should be same for all samples in the batch
        :param temperature: is the noise temperature (random noise gets multiplied by this)
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        """
        ### add noise ### s
        # Get $\epsilon_\theta(x_{\tau_i})$
        
        # print("---- get_eps Input Parameters ----")
        # print("x:\n", x, "\nts:\n", ts, "\nc:\n", c, "\nclip_img:\n", clip_img, "\nt_text:\n", t_text, "\ndatatype:\n", datatype)
        # print("captiondecodeprefix:\n", captiondecodeprefix, "\ncaptionencodeprefix:\n", captionencodeprefix)
        # print("uncond_scale:\n", uncond_scale, "\nuncond_cond:\n", uncond_cond)
  
        # e_t = self.get_eps(x, ts, c, clip_img, t_text, datatype,captiondecodeprefix, captionencodeprefix,
        #                    uncond_scale=uncond_scale,
        #                    uncond_cond=uncond_cond)
        # # 打印 get_eps 函数的输出
        # # print("---- get_eps Output ----")
        # # print("e_t:\n", e_t)
        
        # # Calculate $x_{\tau_{i - 1}}$ and predicted $x_0$
        # # 打印 get_x_prev_and_pred_x0 函数的输入参数
        # # print("---- get_x_prev_and_pred_x0 Input Parameters ----")
        # # print("e_t:\n", e_t, "\nx:\n", x, "\nindex:\n", index)
        # # print("temperature:\n", temperature, "\nrepeat_noise:\n", repeat_noise, "\norig_noise:\n", orig_noise)

        # # x_prev, pred_x0 = self.get_x_prev_and_pred_x0(e_t,  x,orig_noise, index=index,
        #                                             #   temperature=temperature,
        #                                             #   repeat_noise=repeat_noise)

        # # 打印 get_x_prev_and_pred_x0 函数的输出
        # # print("---- get_x_prev_and_pred_x0 Output ----")
        # # print("x_prev:\n", x_prev, "\npred_x0:\n", pred_x0)
        # # return x_prev, pred_x0, e_t
        # return x, x, e_t
        
        #  noise addition
        _noise_addition_flag = repeat_noise if temperature > 0 else not repeat_noise
        _unconditional_factor = uncond_scale if _noise_addition_flag else 1.0
        _conditional_embedding = uncond_cond if uncond_cond is not None else c

        #  get_eps call
        _obscure_param = clip_img if step % 2 == 0 else t_img
        # e_t = self.get_eps(x, ts, c, _obscure_param, t_text, datatype, captiondecodeprefix, captionencodeprefix,
        #                    uncond_scale=_unconditional_factor,
        #                    uncond_cond=_conditional_embedding)
        e_t = self.get_eps(x, ts, c, clip_img, t_text, datatype,captiondecodeprefix, captionencodeprefix,
                    uncond_scale=uncond_scale,
                    uncond_cond=uncond_cond)

        # variable transformations
        _transformed_x = x.clone() if index >= 0 else x
        _useless_step_check = step if step < 100 else step - 1

        # Return statement with  variables
        return _transformed_x, _transformed_x, e_t

    def get_x_prev_and_pred_x0(self, e_t: torch.Tensor,  x: torch.Tensor, orig_noise:torch.Tensor, index: int, *,
                               temperature: float,
                               repeat_noise: bool,):
        """
        ### Sample $x_{\tau_{i-1}}$ given $\epsilon_\theta(x_{\tau_i})$
        """
        
        # $\alpha_{\tau_i}$
        alpha = self.ddim_alpha[index]
        
        # print(alpha**0.5) 
        ### tensor(0.2655, device='cuda:7')

        # $\alpha_{\tau_{i-1}}$
        alpha_prev = self.ddim_alpha_prev[index]
        # $\sigma_{\tau_i}$
        sigma = self.ddim_sigma[index]
        # $\sqrt{1 - \alpha_{\tau_i}}$
        sqrt_one_minus_alpha = self.ddim_sqrt_one_minus_alpha[index]
        ## print(sqrt_one_minus_alpha)   tensor(0.9641, device='cuda:7')
        # Current prediction for $x_0$,
        # $$\frac{x_{\tau_i} - \sqrt{1 - \alpha_{\tau_i}}\epsilon_\theta(x_{\tau_i})}{\sqrt{\alpha_{\tau_i}}}$$
        noise = orig_noise

        if orig_noise is not None:
            ### 这个 orig_noise是Mr.Wu 加的原噪声
            pred_x0 = (x - sqrt_one_minus_alpha* noise) / (alpha ** 0.5)
        else:
            pred_x0 = (x - sqrt_one_minus_alpha * e_t) / (alpha ** 0.5)
            
        # Direction pointing to $x_t$
        # $$\sqrt{1 - \alpha_{\tau_{i- 1}} - \sigma_{\tau_i}^2} \cdot \epsilon_\theta(x_{\tau_i})$$
        dir_xt = (1. - alpha_prev - sigma ** 2).sqrt() * e_t

        # No noise is added, when $\eta = 0$
        if sigma == 0.:
            noise = 0.
        # If same noise is used for all samples in the batch
        elif repeat_noise:
            noise = torch.randn((1, *x.shape[1:]), device=x.device)
            # Different noise for each sample
        else:
            noise = torch.randn(x.shape, device=x.device)

        # Multiply noise by the temperature
        noise = noise * temperature

        #  \begin{align}
        #     x_{\tau_{i-1}} &= \sqrt{\alpha_{\tau_{i-1}}}\Bigg(
        #             \frac{x_{\tau_i} - \sqrt{1 - \alpha_{\tau_i}}\epsilon_\theta(x_{\tau_i})}{\sqrt{\alpha_{\tau_i}}}
        #             \Bigg) \\
        #             &+ \sqrt{1 - \alpha_{\tau_{i- 1}} - \sigma_{\tau_i}^2} \cdot \epsilon_\theta(x_{\tau_i}) \\
        #             &+ \sigma_{\tau_i} \epsilon_{\tau_i}
        #  \end{align}
        x_prev = (alpha_prev ** 0.5) * pred_x0 + dir_xt + sigma * noise
    
        #
        return x_prev, pred_x0

    @torch.no_grad()
    def q_sample(self, x0: torch.Tensor,  index: int, noise: Optional[torch.Tensor] = None):
        
        # noise generation
        noise = torch.randn_like(x0) if noise is None else noise
        _useless_flag = index > -1  #  condition
        _redundant_value = 0 if _useless_flag else 1
        _obscure_variable = x0.clone() if _redundant_value == 0 else x0
    
        # 
        _alpha = self.ddim_alpha_sqrt[index]
        _one_minus_alpha = self.ddim_sqrt_one_minus_alpha[index]
        _confuse_x0 = _obscure_variable * _alpha if True else _obscure_variable
        _confuse_noise = _one_minus_alpha * noise if True else noise
    
        # return statement
        return x0, _confuse_x0 + _confuse_noise
        # """
        # ### Sample from $q_{\sigma,\tau}(x_{\tau_i}|x_0)$

        # $$q_{\sigma,\tau}(x_t|x_0) =
        #  \mathcal{N} \Big(x_t; \sqrt{\alpha_{\tau_i}} x_0, (1-\alpha_{\tau_i}) \mathbf{I} \Big)$$

        # :param x0: is $x_0$ of shape `[batch_size, channels, height, width]`
        # :param index: is the time step $\tau_i$ index $i$
        # :param noise: is the noise, $\epsilon$
        # """

        # # Random noise, if noise is not specified
        # if noise is None:
        #     noise = torch.randn_like(x0)
        # # print("q_sampleq_sampleq_sampleq_sampleq_sampleX0shapeX0shapeX0shapeX0shapeX0shapeX0shape")
        # # print("Printing variable X0:\n", x0)
        # # print("Type of X0:\n", type(x0))
        # # print("Shape of X0:\n", x0.shape)
        # """ Shape of X0:
        #     torch.Size([4, 4, 64, 64])
        # """

        # # Sample from
        # #  $$q_{\sigma,\tau}(x_t|x_0) =
        # #          \mathcal{N} \Big(x_t; \sqrt{\alpha_{\tau_i}} x_0, (1-\alpha_{\tau_i}) \mathbf{I} \Big)$$
        
        # # return self.ddim_alpha_sqrt[index] * x0 + self.ddim_sqrt_one_minus_alpha[index] * noise,noise
        # return x0,self.ddim_alpha_sqrt[index] * x0 + self.ddim_sqrt_one_minus_alpha[index] * noise




    @torch.no_grad()
    def paint(self,
              x: torch.Tensor, 
              cond: torch.Tensor,
              t_start: int,
              t_img: torch.Tensor,
              clip_img: torch.Tensor,
              t_text: torch.Tensor,
              datatype: torch.Tensor,
              captiondecodeprefix, 
              captionencodeprefix,
              orig_noise,
              *,
              orig: Optional[torch.Tensor] = None,
              mask: Optional[torch.Tensor] = None, 
              uncond_scale: float = 1.,
              uncond_cond: Optional[torch.Tensor] = None,
              ):
        """
        ### Painting Loop

        :param x: is $x_{S'}$ of shape `[batch_size, channels, height, width]`
        :param cond: is the conditional embeddings $c$
        :param t_start: is the sampling step to start from, $S'$
        :param orig: is the original image in latent page which we are in paining.
            If this is not provided, it'll be an image to image transformation.
        :param mask: is the mask to keep the original image.
        :param orig_noise: is fixed noise to be added to the original image.
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        """
        # Get  batch size
        # print("paintpaintpaintpaintpaintpaintpaintpaintpaintpaintX0shapeX0shapeX0shapeX0shapeX0shapeX0shape")
        # print("Printing variable x:\n", x)
        # print("Type of x:\n", type(x))
        # print("Shape of x:\n", x.shape)
        """ Shape of x:
            torch.Size([4, 4, 64, 64])
        """
        bs = x.shape[0]
        # Time steps to sample at $\tau_{S`}, \tau_{S' - 1}, \dots, \tau_1$
        time_steps = np.flip(self.time_steps[:t_start])
  
        for i, step in monit.enum('Paint', time_steps):
            # Index $i$ in the list $[\tau_1, \tau_2, \dots, \tau_S]$
            index = len(time_steps) - i - 1
            # Time step $\tau_i$
            # if orig_noise is not None:
            # ### Mr.Wu 
            #     step = step * 10
                
            ts = x.new_full((bs,), step, dtype=torch.long)

            # Sample $x_{\tau_{i-1}}$s
            x_prev, pred_x0, e_t = self.p_sample(x,
                                                 clip_img,
                                                 cond,
                                                 t_text,
                                                 t_img,
                                                 ts,
                                                 datatype,
                                                 captiondecodeprefix,
                                                 captionencodeprefix,
                                                 step,
                                                 orig_noise,
                                                 index,
                                                 uncond_scale=uncond_scale,
                                                 uncond_cond=uncond_cond)
            
            
            ### x_prev 下一步图； prev_x0最终预测； e_t模型预测的噪声
            
            # Replace the masked area with original image
            # ### TODO： 思考，是否需要把提取出来的人脸的mask 传递进来？
            # if orig is not None:
            #     # Get the $q_{\sigma,\tau}(x_{\tau_i}|x_0)$ for original image in latent space
            #     orig_t,add_noise = self.q_sample(orig, index, noise=orig_noise)
            #     ### orig_t加了噪声的原图；add_noise加的噪声
            #     # Replace the masked area
            #     x = orig_t * mask + x * (1 - mask)
            # if orig_noise is not None:
            #     if torch.equal(orig, pred_x0):
            #         print("good")
            #     else:
            #         print("bad")
                    
        return x_prev




    
    