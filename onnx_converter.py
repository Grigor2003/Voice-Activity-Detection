import os
import torch

from other import find_last_model_in_tree

train_res_dir = "train_results"
model_name = r"DGGD_64"

model_trains_tree_dir = os.path.join(train_res_dir, model_name)
model_new_dir, model_path = find_last_model_in_tree(model_trains_tree_dir)
if model_path is None:
    raise Exception(f"No model was found at {model_trains_tree_dir}")
print(f"Model was found at {model_path}")

checkpoint = torch.load(model_path, weights_only=True)


onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)
