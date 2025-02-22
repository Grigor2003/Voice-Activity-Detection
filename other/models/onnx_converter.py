import os

import torch

from other.models.models_handler import MODELS

name = 'DGCGD_64'
example_inp_x = torch.rand(1, 1000, 64)
ckp_path = r"C:\Users\gg\Projects\Voice-Activity-Detection\train_results\DGCGD_64\2025-02-16\res_1\weights.pt"
save_path = os.path.join(os.path.dirname(ckp_path), f'{name}.onnx')
model = MODELS[name]()
checkpoint = torch.load(ckp_path, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])

torch.onnx.export(
    model,
    (example_inp_x,),
    save_path,
    input_names=['x'],
    output_names=['output'],
    dynamic_axes={
        'x': {1: 'seq_len'},
        'output': {1: 'seq_len'}
    }
)
