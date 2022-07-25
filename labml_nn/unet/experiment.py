"""
---
title: Training a U-Net on Carvana dataset
summary: >
  Code for training a U-Net model on Carvana dataset.
---

# Training [U-Net](index.html)

This trains a [U-Net](index.html) model on [Carvana dataset](carvana.html).
You can find the download instructions
[on Kaggle](https://www.kaggle.com/competitions/carvana-image-masking-challenge/data).

Save the training images inside `carvana/train` folder and the masks in `carvana/train_masks` folder.

For simplicity, we do not do a training and validation split.
"""

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms.functional
from torch import nn

from labml import lab, tracker, experiment, monit
from labml.configs import BaseConfigs
from labml_helpers.device import DeviceConfigs
from labml_nn.unet.carvana import CarvanaDataset
from labml_nn.unet import UNet


class Configs(BaseConfigs):
    """
    ## Configurations
    """
    # Device to train the model on.
    # [`DeviceConfigs`](https://docs.labml.ai/api/helpers.html#labml_helpers.device.DeviceConfigs)
    #  picks up an available CUDA device or defaults to CPU.
    device: torch.device = DeviceConfigs()

    # [U-Net](index.html) model
    model: UNet

    # Number of channels in the image. $3$ for RGB.
    image_channels: int = 3
    # Number of channels in the output mask. $1$ for binary mask.
    mask_channels: int = 1

    # Batch size
    batch_size: int = 1
    # Learning rate
    learning_rate: float = 2.5e-4

    # Number of training epochs
    epochs: int = 4

    # Dataset
    dataset: CarvanaDataset
    # Dataloader
    data_loader: torch.utils.data.DataLoader

    # Loss function
    loss_func = nn.BCELoss()
    # Sigmoid function for binary classification
    sigmoid = nn.Sigmoid()

    # Adam optimizer
    optimizer: torch.optim.Adam

    def init(self):
        # Initialize the [Carvana dataset](carvana.html)
        self.dataset = CarvanaDataset(lab.get_data_path() / 'carvana' / 'train',
                                      lab.get_data_path() / 'carvana' / 'train_masks')
        # Initialize the model
        self.model = UNet(self.image_channels, self.mask_channels).to(self.device)

        # Create dataloader
        self.data_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size,
                                                       shuffle=True, pin_memory=True)
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Image logging
        tracker.set_image("sample", True)

    @torch.no_grad()
    def sample(self, idx=-1):
        """
        ### Sample images
        """

        # Get a random sample
        x, _ = self.dataset[np.random.randint(len(self.dataset))]
        # Move data to device
        x = x.to(self.device)

        # Get predicted mask
        mask = self.sigmoid(self.model(x[None, :]))
        # Crop the image to the size of the mask
        x = torchvision.transforms.functional.center_crop(x, [mask.shape[2], mask.shape[3]])
        # Log samples
        tracker.save('sample', x * mask)

    def train(self):
        """
        ### Train for an epoch
        """

        # Iterate through the dataset.
        # Use [`mix`](https://docs.labml.ai/api/monit.html#labml.monit.mix)
        # to sample $50$ times per epoch.
        for _, (image, mask) in monit.mix(('Train', self.data_loader), (self.sample, list(range(50)))):
            # Increment global step
            tracker.add_global_step()
            # Move data to device
            image, mask = image.to(self.device), mask.to(self.device)

            # Make the gradients zero
            self.optimizer.zero_grad()
            # Get predicted mask logits
            logits = self.model(image)
            # Crop the target mask to the size of the logits. Size of the logits will be smaller if we
            # don't use padding in convolutional layers in the U-Net.
            mask = torchvision.transforms.functional.center_crop(mask, [logits.shape[2], logits.shape[3]])
            # Calculate loss
            loss = self.loss_func(self.sigmoid(logits), mask)
            # Compute gradients
            loss.backward()
            # Take an optimization step
            self.optimizer.step()
            # Track the loss
            tracker.save('loss', loss)

    def run(self):
        """
        ### Training loop
        """
        for _ in monit.loop(self.epochs):
            # Train the model
            self.train()
            # New line in the console
            tracker.new_line()
            # Save the model
            experiment.save_checkpoint()


def main():
    # Create experiment
    experiment.create(name='unet')

    # Create configurations
    configs = Configs()

    # Set configurations. You can override the defaults by passing the values in the dictionary.
    experiment.configs(configs, {})

    # Initialize
    configs.init()

    # Set models for saving and loading
    experiment.add_pytorch_models({'model': configs.model})

    # Start and run the training loop
    with experiment.start():
        configs.run()


#
if __name__ == '__main__':
    main()
