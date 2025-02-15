from time import time
import torch
import onnxruntime as ort
from tqdm import tqdm
from other.data.processing import WaveToMFCCConverter
import other.models.models_handler as models_handler


class ModelWithMFCC(torch.nn.Module):
    def __init__(self, mfcc_converter: WaveToMFCCConverter, model):
        super().__init__()
        self.mfcc_converter = mfcc_converter
        self.hop_length = self.mfcc_converter.hop_length
        self.model = model

    def forward(self, waveform):
        mfcc_features = self.mfcc_converter(waveform)[0].unsqueeze(0)
        model_output = self.model(mfcc_features)
        return model_output.squeeze((0, -1))


batch_size = 1
seq_len = 10000
n_mfcc = 64
example_inp = torch.rand(batch_size, seq_len)
ckp_path = ''
model = models_handler.gru_with_denses()
ckp = torch.load(ckp_path)
model.load_state_dict(ckp['model_state_dict'])
mfcc_converter = WaveToMFCCConverter(64, win_length=400, hop_length=200)
full_model = ModelWithMFCC(mfcc_converter=mfcc_converter, model=model)

torch.onnx.export(
    full_model,
    (example_inp,),
    'model.onnx',
    input_names=['waveform'],
    output_names=['output'],
    dynamic_axes={
        'waveform': {0: 'batch_size', 1: 'seq_len'},
        'output': {0: 'batch_size', 1: 'seq_len'}
    },
    dynamo=True
)
