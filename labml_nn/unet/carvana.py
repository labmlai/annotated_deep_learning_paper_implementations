"""
---
title: Carvana dataset for the U-Net experiment
summary: >
  Carvana dataset for the U-Net experiment.
---

# Carvana Dataset for the [U-Net](index.html) [experiment](experiment.html)

You can find the download instructions
[on Kaggle](https://www.kaggle.com/competitions/carvana-image-masking-challenge/data).

Save the training images inside `carvana/train` folder and the masks in `carvana/train_masks` folder.
"""

from torch import nn
from pathlib import Path

import torch.utils.data
import torchvision.transforms.functional
from PIL import Image

from labml import lab


class CarvanaDataset(torch.utils.data.Dataset):
    """
    ## Carvana Dataset
    """

    def __init__(self, image_path: Path, mask_path: Path):
        """
        :param image_path: is the path to the images
        :param mask_path: is the path to the masks
        """
        # Get a dictionary of images by id
        self.images = {p.stem: p for p in image_path.iterdir()}
        # Get a dictionary of masks by id
        self.masks = {p.stem[:-5]: p for p in mask_path.iterdir()}

        # Image ids list
        self.ids = list(self.images.keys())

        # Transformations
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(572),
            torchvision.transforms.ToTensor(),
        ])

    def __getitem__(self, idx: int):
        """
        #### Get an image and its mask.

        :param idx: is index of the image
        """

        # Get image id
        id_ = self.ids[idx]
        # Load image
        image = Image.open(self.images[id_])
        # Transform image and convert it to a PyTorch tensor
        image = self.transforms(image)
        # Load mask
        mask = Image.open(self.masks[id_])
        # Transform mask and convert it to a PyTorch tensor
        mask = self.transforms(mask)

        # The mask values were not $1$, so we scale it appropriately.
        mask = mask / mask.max()

        # Return the image and the mask
        return image, mask

    def __len__(self):
        """
        #### Size of the dataset
        """
        return len(self.ids)


# Testing code
if __name__ == '__main__':
    ds = CarvanaDataset(lab.get_data_path() / 'carvana' / 'train', lab.get_data_path() / 'carvana' / 'train_masks')
