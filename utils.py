import os
from datetime import datetime

from audio_utils import augment_sample
import torch
import torchaudio


class WaveToMFCCConverter:
    def __init__(self, n_mfcc, sample_rate=8000, frame_duration_in_ms=None, win_length=None, hop_length=None):
        self.n_mfcc = n_mfcc

        if frame_duration_in_ms is not None:
            sample_count = torch.tensor(sample_rate * frame_duration_in_ms / 1000, dtype=torch.int)
            win_length = torch.pow(2, torch.ceil(torch.log2(sample_count)).to(torch.int)).to(torch.int).item()
        elif win_length is None:
            return
        win_length = int(win_length)

        if hop_length is None:
            hop_length = int(win_length // 2)
        hop_length = int(hop_length)

        mfcc_params = {
            "n_mfcc": n_mfcc,
            "sample_rate": sample_rate
        }
        mel_params = {
            "n_fft": win_length,
            "win_length": win_length,
            "hop_length": hop_length,
            "center": False
        }

        self.converter = torchaudio.transforms.MFCC(**mfcc_params, melkwargs=mel_params)

    def __call__(self, waveform):
        return self.converter(waveform).transpose(-1, -2)


class NoiseCollate:
    def __init__(self, dataset_sample_rate, noises, params, mfcc_converter):
        self.dataset_sample_rate = dataset_sample_rate
        self.noises = noises
        self.params = params
        self.mfcc_converter = mfcc_converter

    def __call__(self, batch):
        inputs = []
        targets = []

        for au, label_txt in batch:
            au.resample(self.dataset_sample_rate)
            if self.params is None:
                augmented_wave, _ = augment_sample(au, self.noises)
            else:
                augmented_wave, _ = augment_sample(au, self.noises, **self.params)
            inp = self.mfcc_converter(augmented_wave)
            tar = torch.tensor([*map(float, label_txt)])
            if tar.size(-1) != inp.size(-2):
                print(tar.size(-1), inp.size(-2), au.name)
            inputs.append(inp)
            targets.append(tar)

        inp_dim_2 = max(i.size(1) for i in inputs)
        inputs_tens = torch.zeros([len(inputs), inp_dim_2, inputs[0].size(-1)])
        for i, inp in enumerate(inputs):
            inputs_tens[i, :inp.size(1), :] = inp

        tar_dim_1 = max(t.size(0) for t in targets)
        targets_tens = torch.zeros([len(targets), tar_dim_1, 1])
        for i, tar in enumerate(targets):
            targets_tens[i, :tar.size(0), 0] = tar

        return inputs_tens, targets_tens


RES_PREFIX = "res"
ROOT_DIR = "train_results"
DATE_FORMAT = "%Y-%m-%d"
MODEL_NAME = "model.pt"


def find_last_model_in_tree(train_name):

    dir_of_days = os.path.join(ROOT_DIR, train_name)
    res_dir = None

    if os.path.exists(dir_of_days):
        date_objects = [datetime.strptime(date, DATE_FORMAT)
                        for date in os.listdir(dir_of_days)
                        if len(os.listdir(os.path.join(dir_of_days, date))) != 0]
        if len(date_objects) != 0:
            max_num = 0
            day_dir = os.path.join(dir_of_days, max(date_objects).strftime(DATE_FORMAT))
            for name in os.listdir(day_dir):
                st, num = name.split("_")
                folder_path = os.path.join(day_dir, name)
                if max_num <= int(num) and MODEL_NAME in os.listdir(folder_path):
                    max_num = int(num)
                    res_dir = folder_path

    if res_dir is None:
        raise ValueError("No model.pt found")
    else:
        return os.path.join(res_dir, MODEL_NAME)


def create_new_model_dir(train_name):
    day_dir = os.path.join(ROOT_DIR, train_name, datetime.now().strftime(DATE_FORMAT))
    os.makedirs(day_dir, exist_ok=True)
    max_num = 0
    for name in os.listdir(day_dir):
        st, num = name.split("_")
        folder_path = os.path.join(day_dir, name)
        max_num = max(int(num), max_num)
        if len(os.listdir(folder_path)) == 0:
            max_num -= 1
            break

    res_dir = os.path.join(day_dir, RES_PREFIX + "_" + str(max_num + 1))
    os.makedirs(res_dir, exist_ok=True)

    return os.path.join(res_dir, MODEL_NAME)
