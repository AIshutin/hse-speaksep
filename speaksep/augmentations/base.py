from torch import Tensor


class AugmentationBase:
    def __call__(self, data: Tensor) -> Tensor:
        raise NotImplementedError()


class AugmentationBatchBased(AugmentationBase):
    def __init__(self, aug) -> None:
        self._aug = aug
    
    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)