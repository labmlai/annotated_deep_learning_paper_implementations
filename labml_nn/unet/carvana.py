from pathlib import Path


class Dataset:
    def __init__(self, image_path: Path, mask_path: Path):
        self.image_path = image_path
        self.mask_path = mask_path

        names = [p.name for p in self.image_path.iterdir()]
