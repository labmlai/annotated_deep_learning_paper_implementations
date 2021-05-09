"""
---
title: WGAN-GP experiment with MNIST
summary: This experiment generates MNIST images using convolutional neural network.
---

# WGAN-GP experiment with MNIST
"""

import torch

from labml import experiment, tracker
# Import configurations from [Wasserstein experiment](../experiment.html)
from labml_nn.gan.wasserstein.experiment import Configs as OriginalConfigs
#
from labml_nn.gan.wasserstein.gradient_penalty import GradientPenalty


class Configs(OriginalConfigs):
    """
    ## Configuration class

    We extend [original GAN implementation](../../original/experiment.html) and override the discriminator (critic) loss
    calculation to include gradient penalty.
    """

    # Gradient penalty coefficient $\lambda$
    gradient_penalty_coefficient: float = 10.0
    #
    gradient_penalty = GradientPenalty()

    def calc_discriminator_loss(self, data: torch.Tensor):
        """
        This overrides the original discriminator loss calculation and
        includes gradient penalty.
        """
        # Require gradients on $x$ to calculate gradient penalty
        data.requires_grad_()
        # Sample $z \sim p(z)$
        latent = self.sample_z(data.shape[0])
        # $D(x)$
        f_real = self.discriminator(data)
        # $D(G_\theta(z))$
        f_fake = self.discriminator(self.generator(latent).detach())
        # Get discriminator losses
        loss_true, loss_false = self.discriminator_loss(f_real, f_fake)
        # Calculate gradient penalties in training mode
        if self.mode.is_train:
            gradient_penalty = self.gradient_penalty(data, f_real)
            tracker.add("loss.gp.", gradient_penalty)
            loss = loss_true + loss_false + self.gradient_penalty_coefficient * gradient_penalty
        # Skip gradient penalty otherwise
        else:
            loss = loss_true + loss_false

        # Log stuff
        tracker.add("loss.discriminator.true.", loss_true)
        tracker.add("loss.discriminator.false.", loss_false)
        tracker.add("loss.discriminator.", loss)

        return loss


def main():
    # Create configs object
    conf = Configs()
    # Create experiment
    experiment.create(name='mnist_wassertein_gp_dcgan')
    # Override configurations
    experiment.configs(conf,
                       {
                           'discriminator': 'cnn',
                           'generator': 'cnn',
                           'label_smoothing': 0.01,
                           'generator_loss': 'wasserstein',
                           'discriminator_loss': 'wasserstein',
                           'discriminator_k': 5,
                       })

    # Start the experiment and run training loop
    with experiment.start():
        conf.run()


if __name__ == '__main__':
    main()
