from pathlib import Path

import torch.utils.data
import torchvision.transforms.functional
from PIL import Image

from labml import lab


class CarvanaDataset(torch.utils.data.Dataset):
    def __init__(self, image_path: Path, mask_path: Path):
        self.images = {p.stem: p for p in image_path.iterdir()}
        self.masks = {p.stem[:-5]: p for p in mask_path.iterdir()}

        self.ids = list(self.images.keys())

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        image = Image.open(self.images[id_])
        image = torchvision.transforms.functional.to_tensor(image)
        mask = Image.open(self.masks[id_])
        mask = torchvision.transforms.functional.to_tensor(mask)

        return image, mask

    def __len__(self):
        return len(self.ids)


if __name__ == '__main__':
    ds = CarvanaDataset(lab.get_data_path() / 'carvana' / 'train', lab.get_data_path() / 'carvana' / 'train_masks')

    print(ds[0])
