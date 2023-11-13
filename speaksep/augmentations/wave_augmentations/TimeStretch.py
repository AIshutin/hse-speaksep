from torch import Tensor
import torch

from speaksep.augmentations.base import AugmentationBase
from librosa.effects import time_stretch


class TimeStretch(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self.min_rate = kwargs.get('min_rate', 0.9)
        self.max_rate = kwargs.get('max_rate', 1.1)
        self.p = kwargs.get('p', 1.0)
    
    def __call__(self, x: Tensor):
        t = torch.zeros(size=(1,)).uniform_()
        if t > self.p:
            return x
        rate = torch.zeros(size=(1,)).uniform_(self.min_rate, self.max_rate).item()
        x = time_stretch(x.numpy(), rate=rate)
        return torch.tensor(x)