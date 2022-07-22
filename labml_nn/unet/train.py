import numpy as np
import torch
import torch.utils.data
import torchvision.transforms.functional
from torch import nn

from labml import lab, tracker, experiment, monit
from labml.configs import BaseConfigs
from labml_helpers.device import DeviceConfigs
from labml_nn.unet.carvana import CarvanaDataset
from labml_nn.unet.model import UNet


class Configs(BaseConfigs):
    """
    ## Configurations
    """
    # Device to train the model on.
    # [`DeviceConfigs`](https://docs.labml.ai/api/helpers.html#labml_helpers.device.DeviceConfigs)
    #  picks up an available CUDA device or defaults to CPU.
    device: torch.device = DeviceConfigs()

    model: UNet

    # Number of channels in the image. $3$ for RGB.
    image_channels: int = 3
    # Image size
    mask_channels: int = 1

    # Batch size
    batch_size: int = 1
    # Learning rate
    learning_rate: float = 2.5e-4

    # Number of training epochs
    epochs: int = 16

    # Dataset
    dataset: CarvanaDataset
    # Dataloader
    data_loader: torch.utils.data.DataLoader

    loss_func = nn.BCELoss()
    sigmoid = nn.Sigmoid()

    # Adam optimizer
    optimizer: torch.optim.Adam

    def init(self):
        self.dataset = CarvanaDataset(lab.get_data_path() / 'carvana' / 'train',
                                      lab.get_data_path() / 'carvana' / 'train_masks')
        self.model = UNet(self.image_channels, self.mask_channels).to(self.device)

        # Create dataloader
        self.data_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True, pin_memory=True)
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Image logging
        tracker.set_image("sample", True)

    @torch.no_grad()
    def sample(self, idx=-1):
        """
        ### Sample images
        """

        x, _ = self.dataset[np.random.randint(len(self.dataset))]

        x = x.to(self.device)

        mask = self.sigmoid(self.model(x[None, :]))
        x = torchvision.transforms.functional.center_crop(x, [mask.shape[2], mask.shape[3]])
        # Log samples
        tracker.save('sample', x * mask)

    def train(self):
        """
        ### Train
        """

        # Iterate through the dataset
        for _, (image, mask) in monit.mix(('Train', self.data_loader), (self.sample, list(range(50)))):
            # Increment global step
            tracker.add_global_step()
            # Move data to device
            image, mask = image.to(self.device), mask.to(self.device)

            # Make the gradients zero
            self.optimizer.zero_grad()
            pred = self.model(image)
            mask = torchvision.transforms.functional.center_crop(mask, [pred.shape[2], pred.shape[3]])
            # Calculate loss
            loss = self.loss_func(self.sigmoid(pred), mask)
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
    experiment.configs(configs, {
    })

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
