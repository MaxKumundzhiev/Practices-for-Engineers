# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

import random
import numpy as np

import torch
import torchvision.transforms.functional as tvf

from torchvision import transforms
from torchvision.datasets import STL10
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, SequentialSampler


def random_rotate(image):
    if random.random() > 0.5:
        return tvf.rotate(image, angle=random.choice((0, 90, 180, 270)))
    return image


class ResizedRotation:
    def __init__(self, angle, output_size=(96, 96)):
        self.angle = angle
        self.output_size = output_size

    @staticmethod
    def angle_to_rad(ang):
        return np.pi * ang / 100

    def __call__(self, image):
        w, h = image.size
        new_h = int(np.abs(w * np.sin(self.angle_to_rad(90 - self.angle))) + np.abs(h * np.sin(self.angle_to_rad(self.angle))))
        new_w = int(np.abs(h * np.sin(self.angle_to_rad(90 - self.angle))) + np.abs(w * np.sin(self.angle_to_rad(self.angle))))

        img = tvf.resize(image, (new_w, new_h))
        img = tvf.rotate(img, self.angle)
        img = tvf.center_crop(img, self.output_size)
        return img


class WrapWithRandomParams:
    def __init__(self, constructor, ranges):
        self.constructor = constructor
        self.ranges = ranges

    def __call__(self, image):
        randoms = [float(np.random.uniform(low, high)) for _, (low, high) in zip(range(len(self.ranges)), self.ranges)]
        return self.constructor(*randoms)(image)


class PretrainingDatasetWrapper(Dataset):
    def __init__(self, ds: Dataset, target_size=(96,96), debug=False):
        super().__init__()
        self.ds = ds
        self.debug = debug
        self.target_size = target_size
        if debug:
            print("DATASET IN DEBUG MODE")

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        random_resized_rotation = WrapWithRandomParams(lambda angle: ResizedRotation(angle, target_size), [(0.0, 360.0)])
        self.randomize = transforms.Compose([
            transforms.RandomResizedCrop (target_size, scale=(1 / 3, 1.0), ratio=(0.3, 2.0)),
            transforms.RandomChoice([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Lambda(random_rotate)
            ]),
            transforms.RandomApply([
                random_resized_rotation
            ], p=0.33),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2)
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem_internal__(self, idx, preprocess=True):
        this_image_raw, _ = self.ds[idx]

        if self.debug:
            random.seed(idx)
            t1 = self.randomize(this_image_raw)
            random.seed(idx + 1)
            t2 = self.randomize(this_image_raw)
        else:
            t1 = self.randomize(this_image_raw)
            t2 = self.randomize(this_image_raw)

        if preprocess:
            t1 = self.preprocess(t1)
            t2 = self.preprocess(t2)
        else:
            t1 = transforms.ToTensor()(t1)
            t2 = transforms.ToTensor()(t2)

        return (t1, t2), torch.tensor(0)

    def __getitem__(self, idx):
        return self.__getitem_internal__(idx, True)

    def raw(self, idx):
        return self.__getitem_internal__ (idx, False)


stl10_unlabeled = STL10(".", split="unlabeled", download=False)
ds = PretrainingDatasetWrapper(stl10_unlabeled, debug=False)

# Sample visualisation
# import matplotlib.pyplot as plt
# plt.imshow(tvf.to_pil_image(ds[1][0][0])) # augmented
# plt.imshow(tvf.to_pil_image(ds.raw(1)[0][0])) # original
# plt.show()





