import torch

from other.data.processing import WaveToMFCCConverter


class ModelWithMFCC(torch.nn.Module):
    def __init__(self, mfcc_converter: WaveToMFCCConverter, model):
        super().__init__()
        self.mfcc_converter = mfcc_converter
        self.model = model

    def forward(self, waveform):
        mfcc_features = self.mfcc_converter(waveform)[0].unsqueeze(0)
        model_output = self.model(mfcc_features)
        return model_output.squeeze((0, -1))

