from os import path
import unittest
import numpy as np
import torch
from torch.utils.data import SequentialSampler, DataLoader
from torchssim import MSSIM
from utils import (
    ImagePairsDataset,
    RGBToGrayscale,
    ToTensor,
    ComposedPairTransforms,
    make_reproducible,
)


class Tester(unittest.TestCase):
    def test_ssim_cpu(self):
        root = path.join(path.dirname(__file__), "assets", "images")
        transforms = ComposedPairTransforms(RGBToGrayscale(), ToTensor())
        dataset = ImagePairsDataset(root, transforms=transforms)

        make_reproducible()
        sampler = SequentialSampler(dataset)
        data_loader = DataLoader(dataset, batch_size=16, num_workers=1, sampler=sampler)

        mssim = MSSIM().double()
        scores = []
        with torch.no_grad():
            for image1, image2 in data_loader:
                image1, image2 = image1.double(), image2.double()

                score = mssim(image1, image2)
                scores.append(score)

        actual = torch.cat(scores).detach().cpu().numpy()

        # FIXME
        desired = actual

        np.testing.assert_allclose(actual, desired)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    def test_ssim_cuda(self):
        pass
        # FIXME


if __name__ == "__main__":
    unittest.main()
