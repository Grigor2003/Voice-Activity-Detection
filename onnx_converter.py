import os

import torch

from other.models.models_handler import MODELS
from other.utils import find_last_model_in_tree

model_name = 'DGCGD_64'
out_name = model_name + '_music_89' +'.onnx'
example_inp_x = torch.rand(1, 1000, 64)
# ckp_path = r"C:\Users\gg\Projects\Voice-Activity-Detection\train_results\DGCGD_64\2025-03-13\res_1\weights.pt"
_, ckp_path = find_last_model_in_tree(model_name)
print(ckp_path)
save_path = os.path.join(os.path.dirname(ckp_path), out_name)
model = MODELS[model_name]()
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
