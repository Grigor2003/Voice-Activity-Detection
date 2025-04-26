import os

import torch

from other.models.models_handler import MODELS
from other.utils import find_last_model_in_tree

model_name = 'DGCGD_7'
out_name = model_name + '_+6e_noise' + '.onnx'
example_inp_x = torch.rand(1, 1000, 64)
ckp_path = (r"C:\Users\gg\Projects\VAD_infrastructure\Voice-Activity-Detection\RESULTS\DGCGD_7\START_2_(2025-04-25)"
            + r"\run_3\weights.pt")
# _, ckp_path = find_last_model_in_tree(model_name)
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
