import torch.nn as nn

from labml import experiment
from labml.configs import option
from labml_helpers.module import Module
from labml_nn.experiments.cifar10 import CIFAR10Configs
from labml_nn.normalization.batch_norm import BatchNorm


class Configs(CIFAR10Configs):
    pass


class Model(Module):
    """
    ### VGG model for CIFAR-10 classification
    """

    def __init__(self):
        super().__init__()
        layers = []
        # RGB channels
        in_channels = 3
        # Number of channels in each layer in each block
        for block in [[32, 32], [64, 64], [128], [128], [128]]:
            # Convolution, Normalization and Activation layers
            for channels in block:
                layers += [
                    nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
                    BatchNorm(channels, track_running_stats=False),
                    nn.ReLU(inplace=True),
                ]
                in_channels = channels
            # Max pooling at end of each block
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        # Create a sequential model with the layers
        self.layers = nn.Sequential(*layers)
        # Final logits layer
        self.fc = nn.Linear(128, 10)

    def __call__(self, x):
        # The VGG layers
        x = self.layers(x)
        # Reshape for classification layer
        x = x.view(x.shape[0], -1)
        # Final linear layer
        return self.fc(x)


@option(Configs.model)
def _small_model(c: Configs):
    """
    ### Create model
    """
    return Model().to(c.device)


def main():
    # Create experiment
    experiment.create(name='cifar10', comment='small')
    # Create configurations
    conf = Configs()
    # Load configurations
    experiment.configs(conf, {
        'device.cuda_device': 1,
        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 2.5e-4,
    })
    experiment.add_pytorch_models({'model': conf.model})

    # Start the experiment and run the training loop
    with experiment.start():
        conf.run()


#
if __name__ == '__main__':
    main()
