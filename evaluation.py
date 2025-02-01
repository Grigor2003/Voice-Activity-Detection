import numpy as np
import torchaudio
import os
import time

from tqdm import tqdm

import torch

from other import AudioWorker
from models_handler import MODELS
from other import WaveToMFCCConverter
from other import find_last_model_in_tree

input_dir = r"datasets/simple_test/input"
output_dir = r"datasets/simple_test/output"
# model_name = r"WhisperLike_64"
model_name = r"DGGD_64"


train_res_dir = "train_results"

th = 0.72

if __name__ == '__main__':

    model_trains_tree_dir = os.path.join(train_res_dir, model_name)

    model_new_dir, model_path = find_last_model_in_tree(model_trains_tree_dir)

    if model_path is None:
        raise Exception(f"No model was found at {model_trains_tree_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint = torch.load(model_path, weights_only=True)

    sample_rate = checkpoint['mfcc_sample_rate']
    win_length = checkpoint['mfcc_win_length']
    hop_length = checkpoint['mfcc_hop_length']

    model = MODELS[model_name]().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    mfcc_converter = WaveToMFCCConverter(
        n_mfcc=checkpoint['mfcc_n_mfcc'],
        sample_rate=sample_rate,
        win_length=win_length,
        hop_length=hop_length)

    model.eval()

    eval_paths = [os.path.join(input_dir, p) for p in os.listdir(input_dir)
                  if p.endswith(".wav") or p.endswith(".mp3")]

    print(f"{len(eval_paths)} files will be evaluated")
    if len(eval_paths) == 0:
        raise ValueError("No files were found")

    if len(eval_paths) <= 10:
        print("\n".join(eval_paths))

    output_dir = os.path.join(output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    time.sleep(0.5)
    for audio_path in tqdm(eval_paths):
        au = AudioWorker(audio_path, os.path.basename(audio_path))
        au.load()
        au.resample(sample_rate)
        inp = mfcc_converter(au.wave).to(device)
        speech_mask = model(inp).squeeze(-1).detach().cpu().numpy()
        item_wise_mask = np.full(au.wave.size(1), False, dtype=bool)

        for i, speech_lh in enumerate(speech_mask.T):
            item_wise_mask[hop_length * i:hop_length * i + win_length] = (
                    (speech_lh > th) or item_wise_mask[hop_length * i:hop_length * i + win_length])

        torchaudio.save(os.path.join(output_dir, au.name), au.wave[:, item_wise_mask], au.rate)
