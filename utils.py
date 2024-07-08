import os
from datetime import datetime

from tqdm import tqdm

from audio_utils import augment_sample
import torch
import torchaudio


class WaveToMFCCConverter:
    def __init__(self, n_mfcc, sample_rate=8000, frame_duration_in_ms=None, win_length=None, hop_length=None):
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
        self.frame_duration_in_ms = frame_duration_in_ms

        if frame_duration_in_ms is not None:
            sample_count = torch.tensor(sample_rate * frame_duration_in_ms / 1000, dtype=torch.int)
            win_length = torch.pow(2, torch.ceil(torch.log2(sample_count)).to(torch.int)).to(torch.int).item()
        elif win_length is None:
            return
        win_length = int(win_length)

        if hop_length is None:
            hop_length = int(win_length // 2)
        hop_length = int(hop_length)

        self.win_length = win_length
        self.hop_length = hop_length

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


def create_batch_tensor(inputs, targets):
    inp_dim_2 = max(i.size(1) for i in inputs)
    inputs_tens = torch.zeros([len(inputs), inp_dim_2, inputs[0].size(-1)])
    for i, inp in enumerate(inputs):
        inputs_tens[i, :inp.size(1), :] = inp

    tar_dim_1 = max(t.size(0) for t in targets)
    targets_tens = torch.zeros([len(targets), tar_dim_1, 1])
    for i, tar in enumerate(targets):
        targets_tens[i, :tar.size(0), 0] = tar

    return inputs_tens, targets_tens


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

        return create_batch_tensor(inputs, targets)


class ValidationCollate:
    def __init__(self, dataset_sample_rate, noises, params, snr_dbs, mfcc_converter):
        self.dataset_sample_rate = dataset_sample_rate
        self.noises = noises
        self.params = params
        self.snr_dbs = snr_dbs
        self.mfcc_converter = mfcc_converter

    def __call__(self, batch):
        all_inputs = {snr_db: [] for snr_db in self.snr_dbs}
        all_targets = {snr_db: [] for snr_db in self.snr_dbs}

        for au, label_txt in batch:
            au.resample(self.dataset_sample_rate)
            tar = torch.tensor([*map(float, label_txt)])

            for snr_db in self.snr_dbs:
                augmented_wave, _ = augment_sample(au, self.noises, snr_db=snr_db, **self.params)
                inp = self.mfcc_converter(augmented_wave)
                if tar.size(-1) != inp.size(-2):
                    print(tar.size(-1), inp.size(-2), au.name)
                else:
                    all_inputs[snr_db].append(inp)
                    all_targets[snr_db].append(tar)

        all_tensors = {snr_db: None for snr_db in self.snr_dbs}
        for snr_db in self.snr_dbs:
            all_tensors[snr_db] = create_batch_tensor(all_inputs[snr_db], all_targets[snr_db])

        return all_tensors


def get_validation_score(model, loss_function, threshold, snr_dbs, dataloader, device):
    loss = {snr_db: 0.0 for snr_db in snr_dbs}
    correct_count = {snr_db: 0.0 for snr_db in snr_dbs}
    whole_count = {snr_db: 0.0 for snr_db in snr_dbs}
    accuracy = {snr_db: 0.0 for snr_db in snr_dbs}

    for all_tensors in tqdm(dataloader):
        for snr_db in snr_dbs:
            batch_inputs = all_tensors[snr_db][0].to(device)
            batch_targets = all_tensors[snr_db][1].to(device)
            output = model(batch_inputs)
            loss[snr_db] += loss_function(output, batch_targets)
            correct_count[snr_db] += torch.sum((output > threshold) == (batch_targets > threshold))
            whole_count[snr_db] += batch_targets.numel()

    for snr_db in snr_dbs:
        loss[snr_db] /= whole_count[snr_db]
        accuracy[snr_db] = correct_count[snr_db] / whole_count[snr_db]

    return loss, accuracy


RES_PREFIX = "res"
DATE_FORMAT = "%Y-%m-%d"
MODEL_NAME = "model.pt"


def find_last_model_in_tree(model_trains_tree_dir) -> (str, str):
    res_dir = None

    if os.path.exists(model_trains_tree_dir):
        date_objects = [datetime.strptime(date, DATE_FORMAT)
                        for date in os.listdir(model_trains_tree_dir)
                        if len(os.listdir(os.path.join(model_trains_tree_dir, date))) != 0]
        if len(date_objects) != 0:
            max_num = 0
            day_dir = os.path.join(model_trains_tree_dir, max(date_objects).strftime(DATE_FORMAT))
            for name in os.listdir(day_dir):
                st, num = name.split("_")
                folder_path = os.path.join(day_dir, name)
                if max_num <= int(num) and MODEL_NAME in os.listdir(folder_path):
                    max_num = int(num)
                    res_dir = folder_path

    if res_dir is None:
        return None, None
    else:
        return res_dir, os.path.join(res_dir, MODEL_NAME)


def create_new_model_trains_dir(model_trains_tree_dir) -> (str, str):
    day_dir = os.path.join(model_trains_tree_dir, datetime.now().strftime(DATE_FORMAT))
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

    return res_dir, os.path.join(res_dir, MODEL_NAME)
