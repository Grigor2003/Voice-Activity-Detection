import os
import queue
import time

import sounddevice as sd
import soundfile as sf

import torch
import numpy as np

from other.models.models_handler import MODELS
from other.data.processing import WaveToMFCCConverter
from other.utils import find_last_model_in_tree

# model_name = r"WhisperLike_64"
# model_name = r"DGCGD_7"
model_name = r"DGCGCGD_13_7"
th = 0.6
frames_to_pass = 200
orig_filename = "buffer/original_recording.wav"
cropped_filename = "buffer/cropped_recording.wav"

_, model_path = find_last_model_in_tree(model_name)

if model_path is None:
    raise Exception(f"No model was found at {model_name}")
print(f"Model was found at {model_path}")

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


def callback(frame, frame_len, time_info, status):
    """This is called (from a separate thread) for each audio block."""
    q.put(frame.copy())


orig_file = sf.SoundFile(orig_filename, mode='w', samplerate=sample_rate, channels=1)
cropped_file = sf.SoundFile(cropped_filename, mode='w', samplerate=sample_rate, channels=1)
#
# print(sd.query_devices())
# au_device = sd.query_devices(device=28)
# print(*au_device.items(), sep='\n')
# print(*[i for i in checkpoint.items() if len(str(i)) < 100], sep='\n')

try:
    with sd.InputStream(samplerate=sample_rate, blocksize=hop_lenght, channels=1, callback=callback):

        frames = []
        last_frame = None
        last_state = None
        last_frame_pred = None

        while True:
            try:
                half_frame = q.get(block=False)
                orig_file.write(half_frame)

                if last_frame is not None:
                    frame = np.concatenate([last_frame, half_frame], axis=0)
                    wave = torch.from_numpy(frame).squeeze(-1)
                    spectrogram = mfcc_converter(wave).to(device)

                    frames.append(spectrogram)
                    inp_tensor = torch.cat(frames[-frames_to_pass:]).unsqueeze(0)
                    mask = torch.zeros(inp_tensor.size(1)).unsqueeze(0).to(device)
                    st = time.time()
                    out = model(inp_tensor, mask, hidden_state=last_state)
                    pred = out.detach().cpu().numpy().flatten()
                    curr_frame_pred = pred[-1]
                    if len(pred) > 1:
                        last_frame_pred = pred[-2:].mean()

                    try:
                        if len(frames) >= frames_to_pass:
                            last_state = [hs[:, 0, None, :] for hs in model.hidden_states]
                    except AttributeError:
                        pass

                    en = time.time()

                    if last_frame_pred:
                        p = int(last_frame_pred * 100)
                        t = int(th * 100)
                        if p > t:
                            s = t * '|' + (p - t) * '█'
                        else:
                            s = p * '|'

                        print(f"{last_frame_pred >= th:d} {last_frame_pred:.2f}, in {(en - st) * 1000:05.1f} ms - {s}")

                        if last_frame_pred >= th:
                            cropped_file.write(last_frame)

                last_frame = half_frame.copy()

                size = q.qsize()
                if size > 1:
                    print(f"Queued {size} frames")

            except queue.Empty:
                pass
except KeyboardInterrupt:
    orig_file.close()
    cropped_file.close()
    print('\nRecordings saved')
except Exception as e:
    print(type(e).__name__ + ': ' + str(e))
