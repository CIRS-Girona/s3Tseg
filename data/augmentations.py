"""Defines wrappers around several torchvision transforms to appropriately
augment, when applicable, the ground truth masks along with the corresponding
input images.
"""


import numpy
import torch
from PIL import Image

import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor, normalize
from torch.nn.functional import one_hot


class ToTensor():
    def __init__(self, one_hot=None):
        self.one_hot = one_hot
    
    def __call__(self, sample):
        img, mask = sample
        img = to_tensor(img)
        mask = torch.from_numpy(numpy.array(mask)).long()
        if self.one_hot:
            mask = one_hot(mask, num_classes=self.one_hot).permute(2,0,1)
        return (img, mask)


class Normalize():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, sample):
        img, mask = sample
        img = normalize(img, self.mean, self.std)
        return (img, mask)


class RandomRotate():
    def __call__(self, sample):
        img, mask = sample
        degree = numpy.random.choice((-180, -90, 0, 90, 180))
        return (img.rotate(degree), mask.rotate(degree))
     

class RandomHorizontalFlip():
    def __call__(self, sample):
        img, mask = sample
        return (img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)) \
                    if numpy.random.uniform(0,1) > 0.5 else (img, mask)


class RandomVerticalFlip():
    def __call__(self, sample):
        img, mask = sample
        return (img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)) \
                    if numpy.random.uniform(0,1) > 0.5 else (img, mask)


class ColorJitter():
    def __init__(self, p=0.7):
        self.transform = transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
        ], p=p)

    def __call__(self, sample):
        img, mask = sample
        img = self.transform(img)
        return (img, mask)


class GaussianBlur():
    def __init__(self, p=0.5):
        self.transform = transforms.RandomApply([
            transforms.GaussianBlur(3)
        ], p=p)

    def __call__(self, sample):
        img, mask = sample
        img = self.transform(img)
        return (img, mask)


class Sharpen():
    def __init__(self, p=0.5):
        self.transform = transforms.RandomAdjustSharpness(2, p=p)

    def __call__(self, sample):
        img, mask = sample
        img = self.transform(img)
        return (img, mask)
