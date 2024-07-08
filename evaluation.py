import os
import random
import time

import numpy as np
import torchaudio
from tqdm import tqdm
import torch

from audio_utils import AudioWorker, OpenSLRDataset
from gru_model import SimpleG
from utils import NoiseCollate, ValidationCollate, WaveToMFCCConverter
from utils import find_last_model_in_tree, create_new_model_trains_dir, get_validation_score

input_dir = r"data/simple_test/input"
model_path = r""
output_dir = r"data/simple_test/output"
use_last_model_in = r"train_results/SimpleG"

if __name__ == '__main__':

    if not os.path.exists(input_dir):
        raise FileNotFoundError(input_dir)

    _model_dir, _model_path = find_last_model_in_tree(use_last_model_in)
    if _model_path is not None:
        model_path = _model_path

    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    checkpoint = torch.load(model_path)
    global_epoch = checkpoint['epoch']
    sample_rate = checkpoint['mfcc_sample_rate']
    win_lenght = checkpoint['mfcc_win_length']
    hop_lenght = checkpoint['mfcc_hop_length']

    input_size = checkpoint['model_input_size']
    hidden_dim = checkpoint['model_hidden_dim']
    th = 0.5

    print(f"Loaded {model_path} trained to epoch {global_epoch}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleG(input_dim=input_size, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    mfcc_converter = WaveToMFCCConverter(
        n_mfcc=input_size,
        sample_rate=sample_rate,
        win_length=win_lenght,
        hop_length=hop_lenght)

    model.eval()

    eval_paths = [os.path.join(input_dir, p) for p in os.listdir(input_dir)
                  if p.endswith(".wav") or p.endswith(".mp3")]

    print(f"{len(eval_paths)} files will be evaluated")
    if len(eval_paths) == 0:
        raise ValueError("No files were found")

    if len(eval_paths) <= 10:
        print("\n".join(eval_paths))

    os.makedirs(output_dir, exist_ok=True)
    for audio_path in tqdm(eval_paths):
        au = AudioWorker(audio_path, os.path.basename(audio_path))
        au.load()
        au.resample(8000)  # TODO: parse this from mfcc parameters
        inp = mfcc_converter(au.wave).to(device)
        speech_mask = model(inp).squeeze(-1).detach().cpu().numpy()
        item_wise_mask = np.full(au.wave.size(1), False, dtype=bool)

        for i, speech_lh in enumerate(speech_mask.T):
            item_wise_mask[hop_lenght * i:hop_lenght * i + win_lenght] = (
                    (speech_lh > th) or item_wise_mask[hop_lenght * i:hop_lenght * i + win_lenght])

        torchaudio.save(os.path.join(output_dir, au.name), au.wave[:, item_wise_mask], au.rate)
