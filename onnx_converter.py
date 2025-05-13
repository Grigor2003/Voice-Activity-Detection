import os

import torch

from other.models.models_handler import MODELS
from other.utils import find_last_model_in_tree

# model_name = 'DGCGD_7'
model_name = 'DGCGD_41_21_11'
out_name = model_name + '_mega' + '.onnx'
example_inp_x = torch.rand(1, 1000, 64)
# ckp_path = (r"C:\Users\gg\Projects\VAD_infrastructure\Voice-Activity-Detection\RESULTS\DGCGD_7\START_2_(2025-04-25)"
#             + r"\run_3\weights.pt")
# ckp_path = r"C:\Users\gg\Projects\VAD_infrastructure\Voice-Activity-Detection\RESULTS\DGCGCGD_13_7\START_3_(2025-04-27)\run_3_(re3 more zeros with noises)\old\weights_11.pt"
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
