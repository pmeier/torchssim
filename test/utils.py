from abc import ABC
from typing import Optional, Any, Tuple, Callable
from os import path, listdir
from itertools import combinations_with_replacement
import random
import numpy as np
from PIL import Image
import torch
from torchvision.datasets import VisionDataset
from torchvision.transforms import ToTensor as _ToTensor, Grayscale as _RGBToGrayscale


class ImagePairsDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable] = None) -> None:
        super().__init__(root, transforms=transforms)

        image_files = [
            path.join(self.root, file)
            for file in listdir(self.root)
            if file.endswith(".png")
        ]
        self.image_file_pairs = tuple(combinations_with_replacement(image_files, 2))

    def __len__(self) -> int:
        return len(self.image_file_pairs)

    def __getitem__(self, index: int) -> Tuple[Image.Image, Image.Image]:
        image_pair = [Image.open(file) for file in self.image_file_pairs[index]]

        if self.transforms is not None:
            image_pair = self.transforms(*image_pair)

        return image_pair


class PairTransform(ABC):
    def __init__(self, transform: Callable) -> None:
        self.transform = transform

    def __call__(self, image1: Image.Image, image2: Image.Image) -> Tuple[Any, Any]:
        return self.transform(image1), self.transform(image2)


class RGBToGrayscale(PairTransform):
    def __init__(self) -> None:
        super().__init__(_RGBToGrayscale())


class ToTensor(PairTransform):
    def __init__(self) -> None:
        super().__init__(_ToTensor())


class ComposedPairTransforms:
    def __init__(self, *pair_transforms: Callable) -> None:
        self.pair_transforms = pair_transforms

    def __call__(self, image1, image2) -> Tuple[Any, Any]:
        for transform in self.pair_transforms:
            image1, image2 = transform(image1, image2)

        return image1, image2


def make_reproducible(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
