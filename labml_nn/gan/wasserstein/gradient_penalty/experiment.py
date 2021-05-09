"""
---
title: WGAN experiment with MNIST
summary: This experiment generates MNIST images using convolutional neural network.
---

# WGAN experiment with MNIST
"""
from typing import Any

import torch

from labml import experiment, tracker, monit
from labml_helpers.train_valid import BatchIndex
# Import configurations from [Wasserstein experiment](../experiment.html)
from labml_nn.gan.wasserstein.experiment import Configs as OriginalConfigs
from labml_nn.gan.wasserstein.gradient_penalty import GradientPenalty


class Configs(OriginalConfigs):
    gradient_penalty_coefficient: float = 10.0
    gradient_penalty = GradientPenalty()

    def step(self, batch: Any, batch_idx: BatchIndex):
        self.generator.train(self.mode.is_train)
        self.discriminator.train(self.mode.is_train)

        data, target = batch[0].to(self.device), batch[1].to(self.device)
        data.requires_grad_()

        # Increment step in training mode
        if self.mode.is_train:
            tracker.add_global_step(len(data))

        # Train the discriminator
        with monit.section("discriminator"):
            latent = torch.randn(data.shape[0], 100, device=self.device)
            f_real = self.discriminator(data)
            f_fake = self.discriminator(self.generator(latent).detach())
            loss_true, loss_false = self.discriminator_loss(f_real, f_fake)
            if self.mode.is_train:
                gradient_penalty = self.gradient_penalty(data, f_real)
                tracker.add("loss.gp.", gradient_penalty)
                loss = loss_true + loss_false + self.gradient_penalty_coefficient * gradient_penalty
            else:
                loss = loss_true + loss_false

            # Log stuff
            tracker.add("loss.discriminator.true.", loss_true)
            tracker.add("loss.discriminator.false.", loss_false)
            tracker.add("loss.discriminator.", loss)

            # Train
            if self.mode.is_train:
                self.discriminator_optimizer.zero_grad()
                loss.backward()
                if batch_idx.is_last:
                    tracker.add('discriminator', self.discriminator)
                self.discriminator_optimizer.step()

        # Train the generator
        if batch_idx.is_interval(self.discriminator_k):
            with monit.section("generator"):
                latent = torch.randn(data.shape[0], 100, device=self.device)
                generated_images = self.generator(latent)
                f_fake = self.discriminator(generated_images)
                loss = self.generator_loss(f_fake)

                # Log stuff
                tracker.add('generated', generated_images[0:6])
                tracker.add("loss.generator.", loss)

                # Train
                if self.mode.is_train:
                    self.generator_optimizer.zero_grad()
                    loss.backward()
                    if batch_idx.is_last:
                        tracker.add('generator', self.generator)
                    self.generator_optimizer.step()

        tracker.save()


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
