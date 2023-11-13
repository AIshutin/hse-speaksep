import torch
from torch import Tensor
from torch.nn import Module, CrossEntropyLoss
from speaksep.metric.sisdr import SISDR
from torchmetrics.functional.audio.sdr import scale_invariant_signal_distortion_ratio


def sisdr(separated, target_wave) -> Tensor:
        alpha = (target_wave * separated).sum() / target_wave.norm().square()
        nominator = (alpha * target_wave).norm()
        denominator = ((alpha * target_wave - separated).norm() + 1e-6)
        return 20 * torch.log10(nominator / denominator + 1e-6)


class SuperLoss(Module):
    def __init__(self, alpha=0.5, beta=1.0):
        super().__init__()
        self.ce    = CrossEntropyLoss()
        self.alpha = alpha
        self.beta  = beta

    def forward(self, separated, target_wave,
                speaker_id, predicted_speakers,
                is_train=True,
                **batch) -> Tensor:
        if is_train:
            return -self.beta * scale_invariant_signal_distortion_ratio(separated, target_wave).mean() +\
                    self.alpha * self.ce(predicted_speakers, speaker_id)
        else:
            return -self.beta * scale_invariant_signal_distortion_ratio(separated, target_wave).mean()