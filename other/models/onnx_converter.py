import os

import torch

from other.data.processing import WaveToMFCCConverter
from other.models.models_handler import MODELS
from other.models.onnx_utils import ModelWithMFCC

example_inp = torch.rand(1, 10000)
ckp_path = r"C:\Users\gg\Projects\Voice-Activity-Detection\train_results\DGGD_64\2025-02-15\res_1\weights.pt"
save_path = os.path.join(os.path.dirname(ckp_path), 'model.onnx')
model = MODELS['DGGD_64']()
checkpoint = torch.load(ckp_path, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
mfcc_converter = WaveToMFCCConverter(
            n_mfcc=checkpoint['mfcc_n_mfcc'],
            sample_rate=checkpoint['mfcc_sample_rate'],
            win_length=checkpoint['mfcc_win_length'],
            hop_length=checkpoint['mfcc_hop_length'])
full_model = ModelWithMFCC(mfcc_converter=mfcc_converter, model=model)

torch.onnx.export(
    full_model,
    (example_inp,),
    save_path,
    input_names=['waveform'],
    output_names=['output'],
    dynamic_axes={
        'waveform': {0: 'batch_size', 1: 'seq_len'},
        'output': {0: 'batch_size', 1: 'seq_len'}
    },
    dynamo=True
)
