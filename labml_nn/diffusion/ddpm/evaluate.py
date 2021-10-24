"""
---
title: Denoising Diffusion Probabilistic Models (DDPM) evaluation/sampling
summary: >
  Code to generate samples from a trained
  Denoising Diffusion Probabilistic Model.
---

# [Denoising Diffusion Probabilistic Models (DDPM)](index.html) evaluation/sampling

This is the code to generate images and create interpolations between given images.
"""

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms.functional import to_pil_image, resize

from labml import experiment, monit
from labml_nn.diffusion.ddpm import DenoiseDiffusion, gather
from labml_nn.diffusion.ddpm.experiment import Configs


class Sampler:
    """
    ## Sampler class
    """

    def __init__(self, diffusion: DenoiseDiffusion, image_channels: int, image_size: int, device: torch.device):
        """
        * `diffusion` is the `DenoiseDiffusion` instance
        * `image_channels` is the number of channels in the image
        * `image_size` is the image size
        * `device` is the device of the model
        """
        self.device = device
        self.image_size = image_size
        self.image_channels = image_channels
        self.diffusion = diffusion

        # $T$
        self.n_steps = diffusion.n_steps
        # $\textcolor{cyan}{\epsilon_\theta}(x_t, t)$
        self.eps_model = diffusion.eps_model
        # $\beta_t$
        self.beta = diffusion.beta
        # $\alpha_t$
        self.alpha = diffusion.alpha
        # $\bar\alpha_t$
        self.alpha_bar = diffusion.alpha_bar
        # $\bar\alpha_{t-1}$
        alpha_bar_tm1 = torch.cat([self.alpha_bar.new_ones((1,)), self.alpha_bar[:-1]])

        # To calculate
        #
        # \begin{align}
        # q(x_{t-1}|x_t, x_0) &= \mathcal{N} \Big(x_{t-1}; \tilde\mu_t(x_t, x_0), \tilde\beta_t \mathbf{I} \Big) \\
        # \tilde\mu_t(x_t, x_0) &= \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}x_0
        #                          + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1-\bar\alpha_t}x_t \\
        # \tilde\beta_t &= \frac{1 - \bar\alpha_{t-1}}{a}
        # \end{align}

        # $\tilde\beta_t$
        self.beta_tilde = self.beta * (1 - alpha_bar_tm1) / (1 - self.alpha_bar)
        # $$\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}$$
        self.mu_tilde_coef1 = self.beta * (alpha_bar_tm1 ** 0.5) / (1 - self.alpha_bar)
        # $$\frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1}}{1-\bar\alpha_t}$$
        self.mu_tilde_coef2 = (self.alpha ** 0.5) * (1 - alpha_bar_tm1) / (1 - self.alpha_bar)
        # $\sigma^2 = \beta$
        self.sigma2 = self.beta

    def show_image(self, img, title=""):
        """Helper function to display an image"""
        img = img.clip(0, 1)
        img = img.cpu().numpy()
        plt.imshow(img.transpose(1, 2, 0))
        plt.title(title)
        plt.show()

    def make_video(self, frames, path="video.mp4"):
        """Helper function to create a video"""
        import imageio
        # 20 second video
        writer = imageio.get_writer(path, fps=len(frames) // 20)
        # Add each image
        for f in frames:
            f = f.clip(0, 1)
            f = to_pil_image(resize(f, [368, 368]))
            writer.append_data(np.array(f))
        #
        writer.close()

    def sample_animation(self, n_frames: int = 1000, create_video: bool = True):
        """
        #### Sample an image step-by-step using $\textcolor{cyan}{p_\theta}(x_{t-1}|x_t)$

        We sample an image step-by-step using $\textcolor{cyan}{p_\theta}(x_{t-1}|x_t)$ and at each step
        show the estimate
        $$x_0 \approx \hat{x}_0 = \frac{1}{\sqrt{\bar\alpha}}
         \Big( x_t - \sqrt{1 - \bar\alpha_t} \textcolor{cyan}{\epsilon_\theta}(x_t, t) \Big)$$
        """

        # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
        xt = torch.randn([1, self.image_channels, self.image_size, self.image_size], device=self.device)

        # Interval to log $\hat{x}_0$
        interval = self.n_steps // n_frames
        # Frames for video
        frames = []
        # Sample $T$ steps
        for t_inv in monit.iterate('Denoise', self.n_steps):
            # $t$
            t_ = self.n_steps - t_inv - 1
            # $t$ in a tensor
            t = xt.new_full((1,), t_, dtype=torch.long)
            # $\textcolor{cyan}{\epsilon_\theta}(x_t, t)$
            eps_theta = self.eps_model(xt, t)
            if t_ % interval == 0:
                # Get $\hat{x}_0$ and add to frames
                x0 = self.p_x0(xt, t, eps_theta)
                frames.append(x0[0])
                if not create_video:
                    self.show_image(x0[0], f"{t_}")
            # Sample from $\textcolor{cyan}{p_\theta}(x_{t-1}|x_t)$
            xt = self.p_sample(xt, t, eps_theta)

        # Make video
        if create_video:
            self.make_video(frames)

    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, lambda_: float, t_: int = 100):
        """
        #### Interpolate two images $x_0$ and $x'_0$

        We get $x_t \sim q(x_t|x_0)$ and $x'_t \sim q(x'_t|x_0)$.

        Then interpolate to
         $$\bar{x}_t = (1 - \lambda)x_t + \lambda x'_0$$

        Then get
         $$\bar{x}_0 \sim \textcolor{cyan}{p_\theta}(x_0|\bar{x}_t)$$

        * `x1` is $x_0$
        * `x2` is $x'_0$
        * `lambda_` is $\lambda$
        * `t_` is $t$
        """

        # Number of samples
        n_samples = x1.shape[0]
        # $t$ tensor
        t = torch.full((n_samples,), t_, device=self.device)
        # $$\bar{x}_t = (1 - \lambda)x_t + \lambda x'_0$$
        xt = (1 - lambda_) * self.diffusion.q_sample(x1, t) + lambda_ * self.diffusion.q_sample(x2, t)

        # $$\bar{x}_0 \sim \textcolor{cyan}{p_\theta}(x_0|\bar{x}_t)$$
        return self._sample_x0(xt, t_)

    def interpolate_animate(self, x1: torch.Tensor, x2: torch.Tensor, n_frames: int = 100, t_: int = 100,
                            create_video=True):
        """
        #### Interpolate two images $x_0$ and $x'_0$ and make a video

        * `x1` is $x_0$
        * `x2` is $x'_0$
        * `n_frames` is the number of frames for the image
        * `t_` is $t$
        * `create_video` specifies whether to make a video or to show each frame
        """

        # Show original images
        self.show_image(x1, "x1")
        self.show_image(x2, "x2")
        # Add batch dimension
        x1 = x1[None, :, :, :]
        x2 = x2[None, :, :, :]
        # $t$ tensor
        t = torch.full((1,), t_, device=self.device)
        # $x_t \sim q(x_t|x_0)$
        x1t = self.diffusion.q_sample(x1, t)
        # $x'_t \sim q(x'_t|x_0)$
        x2t = self.diffusion.q_sample(x2, t)

        frames = []
        # Get frames with different $\lambda$
        for i in monit.iterate('Interpolate', n_frames + 1, is_children_silent=True):
            # $\lambda$
            lambda_ = i / n_frames
            # $$\bar{x}_t = (1 - \lambda)x_t + \lambda x'_0$$
            xt = (1 - lambda_) * x1t + lambda_ * x2t
            # $$\bar{x}_0 \sim \textcolor{cyan}{p_\theta}(x_0|\bar{x}_t)$$
            x0 = self._sample_x0(xt, t_)
            # Add to frames
            frames.append(x0[0])
            # Show frame
            if not create_video:
                self.show_image(x0[0], f"{lambda_ :.2f}")

        # Make video
        if create_video:
            self.make_video(frames)

    def _sample_x0(self, xt: torch.Tensor, n_steps: int):
        """
        #### Sample an image using $\textcolor{cyan}{p_\theta}(x_{t-1}|x_t)$

        * `xt` is $x_t$
        * `n_steps` is $t$
        """

        # Number of sampels
        n_samples = xt.shape[0]
        # Iterate until $t$ steps
        for t_ in monit.iterate('Denoise', n_steps):
            t = n_steps - t_ - 1
            # Sample from $\textcolor{cyan}{p_\theta}(x_{t-1}|x_t)$
            xt = self.diffusion.p_sample(xt, xt.new_full((n_samples,), t, dtype=torch.long))

        # Return $x_0$
        return xt

    def sample(self, n_samples: int = 16):
        """
        #### Generate images
        """
        # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
        xt = torch.randn([n_samples, self.image_channels, self.image_size, self.image_size], device=self.device)

        # $$x_0 \sim \textcolor{cyan}{p_\theta}(x_0|x_t)$$
        x0 = self._sample_x0(xt, self.n_steps)

        # Show images
        for i in range(n_samples):
            self.show_image(x0[i])

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, eps_theta: torch.Tensor):
        """
        #### Sample from $\textcolor{cyan}{p_\theta}(x_{t-1}|x_t)$

        \begin{align}
        \textcolor{cyan}{p_\theta}(x_{t-1} | x_t) &= \mathcal{N}\big(x_{t-1};
        \textcolor{cyan}{\mu_\theta}(x_t, t), \sigma_t^2 \mathbf{I} \big) \\
        \textcolor{cyan}{\mu_\theta}(x_t, t)
          &= \frac{1}{\sqrt{\alpha_t}} \Big(x_t -
            \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{cyan}{\epsilon_\theta}(x_t, t) \Big)
        \end{align}
        """
        # [gather](utils.html) $\bar\alpha_t$
        alpha_bar = gather(self.alpha_bar, t)
        # $\alpha_t$
        alpha = gather(self.alpha, t)
        # $\frac{\beta}{\sqrt{1-\bar\alpha_t}}$
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
        #      \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{cyan}{\epsilon_\theta}(x_t, t) \Big)$$
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        # $\sigma^2$
        var = gather(self.sigma2, t)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        eps = torch.randn(xt.shape, device=xt.device)
        # Sample
        return mean + (var ** .5) * eps

    def p_x0(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):
        """
        #### Estimate $x_0$

        $$x_0 \approx \hat{x}_0 = \frac{1}{\sqrt{\bar\alpha}}
         \Big( x_t - \sqrt{1 - \bar\alpha_t} \textcolor{cyan}{\epsilon_\theta}(x_t, t) \Big)$$
        """
        # [gather](utils.html) $\bar\alpha_t$
        alpha_bar = gather(self.alpha_bar, t)

        # $$x_0 \approx \hat{x}_0 = \frac{1}{\sqrt{\bar\alpha}}
        #  \Big( x_t - \sqrt{1 - \bar\alpha_t} \textcolor{cyan}{\epsilon_\theta}(x_t, t) \Big)$$
        return (xt - (1 - alpha_bar) ** 0.5 * eps) / (alpha_bar ** 0.5)


def main():
    """Generate samples"""

    # Training experiment run UUID
    run_uuid = "a44333ea251411ec8007d1a1762ed686"

    # Start an evaluation
    experiment.evaluate()

    # Create configs
    configs = Configs()
    # Load custom configuration of the training run
    configs_dict = experiment.load_configs(run_uuid)
    # Set configurations
    experiment.configs(configs, configs_dict)

    # Initialize
    configs.init()

    # Set PyTorch modules for saving and loading
    experiment.add_pytorch_models({'eps_model': configs.eps_model})

    # Load training experiment
    experiment.load(run_uuid)

    # Create sampler
    sampler = Sampler(diffusion=configs.diffusion,
                      image_channels=configs.image_channels,
                      image_size=configs.image_size,
                      device=configs.device)

    # Start evaluation
    with experiment.start():
        # No gradients
        with torch.no_grad():
            # Sample an image with an denoising animation
            sampler.sample_animation()

            if False:
                # Get some images fro data
                data = next(iter(configs.data_loader)).to(configs.device)

                # Create an interpolation animation
                sampler.interpolate_animate(data[0], data[1])


#
if __name__ == '__main__':
    main()
