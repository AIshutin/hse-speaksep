from torch import nn
from torch.nn import Sequential
import torch


class PlaceHolderModel(nn.Module):
    def __init__(self, n_classes, width, **kwargs):
        super().__init__()
        self.speaker = nn.Linear(width, n_classes)
        self.width = width
    
    def forward(self, mixed_wave, **batch):
        preds = self.speaker(mixed_wave[:, :self.width])

        return {
            "separated": mixed_wave,
            "predicted_speakers": preds
        }