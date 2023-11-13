from torch import Tensor
import torch

from speaksep.augmentations.base import AugmentationBase
from audiomentations import AddGaussianSNR

class AddGaussianNoiseSNR(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self.rate = kwargs["sample_rate"]
        kwargs.pop("sample_rate")
        self._aug = AddGaussianSNR(*args, **kwargs)
    
    def __call__(self, data: Tensor):
        return torch.tensor(self._aug(data.numpy(), sample_rate=self.rate))