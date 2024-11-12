import os
import queue
import time

import numpy as np
import sounddevice as sd
import torch

from models_handler import MODELS
from other.utils import find_last_model_in_tree, WaveToMFCCConverter

model_name = r"WhisperLike_64"
# model_name = r"DGGD_64"
train_res_dir = "train_results"
th = 0.75

model_trains_tree_dir = os.path.join(train_res_dir, model_name)

model_new_dir, model_path = find_last_model_in_tree(model_trains_tree_dir)

if model_path is None:
    raise Exception(f"No model was found at {model_trains_tree_dir}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

checkpoint = torch.load(model_path, weights_only=True)

sample_rate = checkpoint['mfcc_sample_rate']
win_lenght = checkpoint['mfcc_win_length']
hop_lenght = checkpoint['mfcc_hop_length']

print("sample_rate:", sample_rate)
if 2 * hop_lenght != win_lenght:
    raise Exception("works only when the hop_lenght is half of the win_lenght")

model = MODELS[model_name]().to(device)
model.load_state_dict(checkpoint['model_state_dict'])

mfcc_converter = WaveToMFCCConverter(
    n_mfcc=checkpoint['mfcc_n_mfcc'],
    sample_rate=sample_rate,
    win_length=win_lenght,
    hop_length=hop_lenght)

model.eval()

q = queue.Queue()


def callback(overlap_frame, frame_len, time_info, status):
    """This is called (from a separate thread) for each audio block."""
    q.put(overlap_frame.copy())


frames = []
last_frame = None

with sd.InputStream(samplerate=sample_rate, blocksize=hop_lenght, channels=1, callback=callback):
    while True:
        try:
            overlap_frame = q.get(block=False)

            if last_frame is not None:
                frame = np.concatenate([last_frame, overlap_frame], axis=0)
                wave = torch.from_numpy(frame).squeeze(-1)
                spectrogram = mfcc_converter(wave).to(device)

                frames.append(spectrogram)
                inp_tensor = torch.cat(frames[-200:]).unsqueeze(0)
                st = time.time()
                out = model(inp_tensor).detach().cpu().numpy().flatten()[-1]
                en = time.time()

                p = int(out * 100)
                t = int(th * 100)
                if p > t:
                    s = t * '|' + (p - t) * '█'
                else:
                    s = p * '|'

                print(f"{out >= th:d} {out:05.2f}, in {(en - st) * 1000:05.2f} ms - {s}")

            last_frame = overlap_frame.copy()

            size = q.qsize()
            if size > 1:
                print(f"Queued {size} frames")

        except queue.Empty:
            pass
