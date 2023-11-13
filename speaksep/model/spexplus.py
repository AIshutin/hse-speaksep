from torch import nn
from torch.nn import Sequential
import torch


class SpeechEncoder(nn.Module):
    def __init__(self, n_filters, l1, l2, l3) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, n_filters, kernel_size=l1, stride=l1 // 2, padding=l1 // 2)
        self.conv2 = nn.Conv1d(1, n_filters, kernel_size=l2, stride=l1 // 2, padding=l2 // 2)
        self.conv3 = nn.Conv1d(1, n_filters, kernel_size=l3, stride=l1 // 2, padding=l3 // 2)
        self.activ = nn.ReLU()
    
    def forward(self, X):
        X = X.unsqueeze(1)
        X1 = self.conv1(X)
        X2 = self.conv2(X)
        X3 = self.conv3(X)
        assert(len(X3.shape) == 3)
        X1 = self.activ(X1)
        X2 = self.activ(X2)
        X3 = self.activ(X3)
        X_new = torch.cat((X1, X2, X3), dim=1)
        return X_new, X1, X2, X3


class ResNetBlock(nn.Module):
    def __init__(self, C) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(C, C, kernel_size=1)
        self.bn_activ = nn.Sequential(
            nn.BatchNorm1d(C),
            nn.PReLU()
        )
        self.conv2 = nn.Conv1d(C, C, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(C)
        self.activ2 = nn.PReLU()
        self.pool = nn.MaxPool1d(kernel_size=3)
    
    def forward(self, X):
        X2 = self.conv1(X)
        X2 = self.bn_activ(X2)
        X2 = self.conv2(X2)
        X2 = self.bn2(X2)
        X = X + X2
        X = self.activ2(X)
        return self.pool(X)


class SpeakerEncoder(nn.Module):
    def __init__(self, C_in, C, R_n, n_classes) -> None:
        super().__init__()
        self.norm   = ChanneledLayerNorm(C_in)
        self.conv1  = nn.Conv1d(C_in, C, kernel_size=1)
        self.resnet = nn.Sequential(
            *[ResNetBlock(C) for i in range(R_n)]
        )
        self.conv2 = nn.Conv1d(C, C, kernel_size=1)
        self.head = nn.Linear(C, n_classes)
    
    def forward(self, X):
        X = self.norm(X)
        X = self.conv1(X)
        X = self.resnet(X)
        X = self.conv2(X)
        embeddings = X.mean(dim=-1)
        logits = self.head(embeddings)
        return embeddings, logits


class SimpleModel(nn.Module):
    def __init__(self, n_classes, l1, l2, l3, n_filters, R_n, **kwargs):
        super().__init__()
        self.speech_encoder = SpeechEncoder(n_filters, l1, l2, l3)
        self.speaker_encoder = SpeakerEncoder(n_filters * 3, R_n, n_classes)
    
    def forward(self, mixed_wave, ref_wave, **batch):
        X, Y1, Y2, Y3 = self.speech_encoder(ref_wave)
        embeddings, preds = self.speaker_encoder(X)

        return {
            "separated": mixed_wave,
            "predicted_speakers": preds
        }


class DepthWiseSeparableConvolution(nn.Module):
    # https://arxiv.org/pdf/1704.04861v1.pdf

    def __init__(self, in_channels, out_channels=None, kernel_size=1, **kwargs) -> None:
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.conv_long = nn.Conv1d(in_channels, out_channels, kernel_size, 
                                   groups=in_channels, **kwargs)
        self.conv_short = nn.Conv1d(out_channels, out_channels, kernel_size=1, **kwargs)
    
    def forward(self, X):
        X = self.conv_long(X)
        return self.conv_short(X)


class ChanneledLayerNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.ln = nn.LayerNorm(*args, **kwargs)

    def forward(self, X):
        return self.ln(X.transpose(1, 2)).transpose(1, 2)
    

class GlobalLayerNorm(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.lin = nn.Linear(1, C)
        # I can do something about saving mean and std for inference
        # ...
        # but it's not needed in our case, since inference is batched

    def forward(self, X):
        # B, C, L = X.shape
        mean = X.mean(dim=(1, 2), keepdim=True)
        std = (X - mean).std(dim=(1, 2), keepdim=True)
        X_norm = (X - mean) / std
        return X_norm * self.lin.weight.view(1, -1, 1) + self.lin.bias.view(1, -1, 1)


class TCNBlock(nn.Module):
    def __init__(self, C, kernel_size=3, C_embed=0, dilation=1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(C + C_embed, C + C_embed, kernel_size=1),
            nn.PReLU(),
            GlobalLayerNorm(C + C_embed),
            DepthWiseSeparableConvolution(C + C_embed, C + C_embed, 
                                          kernel_size=kernel_size,padding='same', 
                                          dilation=dilation),
            nn.PReLU(),
            GlobalLayerNorm(C + C_embed),
            nn.Conv1d(C + C_embed, C, kernel_size=1)
        )
        self.C = C

    def forward(self, X, embeddings=None):
        X_init = X.clone()
        if embeddings is not None:
            X = torch.cat((X, embeddings.unsqueeze(-1).repeat(1, 1, X.shape[-1])), dim=1)
        X = self.cnn(X)
        X = X_init + X
        return X
    

class TCNNet(nn.Module):
    def __init__(self, C, C_embed, kernel_size, dilations):
        super().__init__()
        self.net = nn.ModuleList(
            TCNBlock(C, kernel_size, C_embed * (i == 0), dilation=dilation) 
              for i, dilation in enumerate(dilations)
        )
    
    def forward(self, X, embeddings):
        for i, layer in enumerate(self.net):
            X = layer(X, embeddings if i == 0 else None)
        return X


class SpeechExtractor(nn.Module):
    def __init__(self, C, C_out, C_embed, n_stacks, tcn_kernel_size, dilations, l1, l2, l3) -> None:
        super().__init__()
        # print('C', C)
        self.norm = ChanneledLayerNorm(C)
        self.conv1 = nn.Conv1d(C, C, kernel_size=1)
        self.tcn_stacks = nn.ModuleList(
            TCNNet(C, C_embed, tcn_kernel_size, dilations) for i in range(n_stacks)
        )
        self.l1_branch = nn.Sequential(
            nn.Conv1d(C, C_out, kernel_size=l1, padding='same'),
            nn.ReLU()
        )
        self.l2_branch = nn.Sequential(
            nn.Conv1d(C, C_out, kernel_size=l2, padding='same'),
            nn.ReLU()
        )
        self.l3_branch = nn.Sequential(
            nn.Conv1d(C, C_out, kernel_size=l3, padding='same'),
            nn.ReLU()
        )
    
    def forward(self, X, embeddings):
        X = self.norm(X)
        X = self.conv1(X)
        for el in self.tcn_stacks:
            X = el(X, embeddings)
        X1 = self.l1_branch(X)
        X2 = self.l2_branch(X)
        X3 = self.l3_branch(X)
        return X1, X2, X3
        

class SpeechDecoder(nn.Module):
    def __init__(self, C, kernel_size, stride) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose1d(C, 1, kernel_size=kernel_size,
                                               stride=stride, 
                                               padding=stride)
    
    def forward(self, X):
        return self.deconv(X)


class SpexPlus(nn.Module):
    def __init__(self, n_classes, l1, l2, l3, C_embed, speech_encoder, speaker_encoder, \
                 speaker_extractor) -> None:
        super().__init__()

        C = speech_encoder['n_filters'] * 3

        self.speech_encoder = SpeechEncoder(**speech_encoder, l1=l1, l2=l2, l3=l3)        
        self.speaker_encoder = SpeakerEncoder(n_classes=n_classes, C_in=C, C=C_embed, **speaker_encoder)
        self.speech_extractor = SpeechExtractor(C=C, C_out=speech_encoder['n_filters'], C_embed=C_embed,
                                                l1=l1, l2=l2, l3=l3, **speaker_extractor)
        self.speech_decoder = nn.ModuleList(SpeechDecoder(C=speech_encoder['n_filters'], kernel_size=el,
                                                          stride=l1 // 2) 
                                            for el in [l1, l2, l3])

    def forward(self, mixed_wave, ref_wave, **batch):
        ref_encoded, _, _, _ = self.speech_encoder(ref_wave)
        embeddings, preds = self.speaker_encoder(ref_encoded)
        X, Y1, Y2, Y3 = self.speech_encoder(mixed_wave)
        Xs = self.speech_extractor(X, embeddings)
        assert(isinstance(Xs, tuple))
        Ys = [Y1, Y2, Y3]
        decoded = [self.speech_decoder[i](Xs[i] * Ys[i]) for i in range(3)]

        for i in range(len(decoded)):
            decoded[i] = nn.functional.interpolate(decoded[i], size=mixed_wave.shape[-1]).squeeze(1)
        decoded = sum(decoded)

        return {
            "separated": decoded,
            "predicted_speakers": preds
        }