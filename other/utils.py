import os
import random
from datetime import datetime

from matplotlib import pyplot as plt
from torch.nn.utils.rnn import pad_sequence

from other.audio_utils import AudioWorker
from other.augmentation_utils import augment_sample, generate_white_noise
import torch
from torch.utils.data import DataLoader, random_split
import torchaudio
from tabulate import tabulate
import numpy as np

from other.work_with_stamps_utils import stamps_to_binary_counts

RES_PREFIX = "res"
DATE_FORMAT = "%Y-%m-%d"
MODEL_NAME = "weights.pt"
EXAMPLE_FOLDER = "examples"


def get_train_val_dataloaders(dataset, train_ratio, batch_size, val_batch_size, num_workers, val_num_workers,
                              seed=None):
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    if seed is None:
        seed = torch.randint(low=0, high=2 ** 32, size=(1,)).item()
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=val_num_workers)
    return train_dataloader, val_dataloader, seed


class WaveToMFCCConverter:
    def __init__(self, n_mfcc, sample_rate=8000, frame_duration_in_ms=None, win_length=None, hop_length=None):
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
        self.frame_duration_in_ms = frame_duration_in_ms

        if frame_duration_in_ms is not None:
            sample_count = torch.tensor(sample_rate * frame_duration_in_ms / 1000, dtype=torch.int)
            win_length = torch.pow(2, torch.ceil(torch.log2(sample_count)).to(torch.int)).to(torch.int).item()
        elif win_length is None:
            win_length = sample_rate // 20
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
    lengths = [t.size(0) for t in inputs]
    max_len = max(lengths)
    mask = torch.arange(max_len).expand(len(lengths), max_len) < torch.tensor(lengths).unsqueeze(1)
    padded_input = pad_sequence(inputs, batch_first=True)
    padded_output = pad_sequence(targets, batch_first=True)

    return padded_input, mask, padded_output


class NoiseCollate:
    def __init__(self, sample_rate, params, snr_dbs_dict, mfcc_converter, zero_sample_count=0):
        self.sample_rate = sample_rate
        self.noises = None
        self.params = params
        self.snr_dbs = []
        for snr, freq in snr_dbs_dict.items():
            self.snr_dbs.extend([snr] * freq)
        self.mfcc_converter = mfcc_converter
        self.zsc = zero_sample_count

    def __call__(self, batch):
        # TODO:
        # if self.zsc > 0:
        #     sizes = [(i.wave.size(-1), len(t), i.rate) for i, t in batch]
        #
        #     for i in range(self.zsc):
        #         size, t_size, sr = random.choice(sizes)
        #         au = AudioWorker.from_wave(generate_white_noise(1, size, 0.15, 0.1), sr)
        #         batch.append((au, "0" * t_size))

        inputs, targets, examples = [], [], []
        ex_id = random.randint(1, len(batch) - 2) if len(batch) > 2 else None
        for i, (au, label_stamps) in enumerate(batch):
            # au.resample(self.sample_rate)
            tar = torch.tensor(stamps_to_binary_counts(label_stamps, au.wave.size(-1)))

            snr_db = random.choice(self.snr_dbs)

            augmented_wave, _ = augment_sample(au, self.noises, snr_db=snr_db, **self.params)
            inp = self.mfcc_converter(augmented_wave)

            if tar.size(-1) != inp.size(-2):
                print(f"WARNING: mismatch of target {tar.size(-1)} and input {inp.size(-2)} sizes in {au.name}")
            else:
                inputs.append(inp.squeeze(0))
                targets.append(tar)
            if i == ex_id or i == 0 or i == len(batch) - 1:
                examples.append((i, augmented_wave.clone(), f"noise_snr_{snr_db}"))
        return create_batch_tensor(inputs, targets), examples


class ValCollate:
    def __init__(self, sample_rate, params, snr_dbs, mfcc_converter):
        self.sample_rate = sample_rate
        self.noises = None
        self.params = params
        self.snr_dbs = snr_dbs
        self.mfcc_converter = mfcc_converter

    def __call__(self, batch):
        all_inputs = {snr_db: [] for snr_db in self.snr_dbs}
        all_targets = {snr_db: [] for snr_db in self.snr_dbs}

        for au, label_txt in batch:
            au.resample(self.sample_rate)
            tar = torch.tensor([*map(float, label_txt)])

            for snr_db in self.snr_dbs:
                augmented_wave, _ = augment_sample(au, self.noises, snr_db=snr_db, **self.params)
                inp = self.mfcc_converter(augmented_wave)
                if tar.size(-1) != inp.size(-2):
                    print(f"WARNING: mismatch of target {tar.size(-1)} and input {inp.size(-2)} sizes in {au.name}")
                else:
                    all_inputs[snr_db].append(inp.squeeze(0))
                    all_targets[snr_db].append(tar)

        return {snr_db: create_batch_tensor(all_inputs[snr_db], all_targets[snr_db]) for snr_db in self.snr_dbs}


def focal_loss(pred, target, gamma=2., alpha=0.5, reduction="sum"):
    p_t = pred * target + (1 - pred) * (1 - target)
    ce_loss = torch.nn.functional.binary_cross_entropy(pred, target, reduction='none')
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * loss
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss


def print_as_table(dataframe):
    if len(dataframe) > 4:
        print(tabulate(dataframe.iloc[[0, -3, -2, -1], :].T.fillna("---"), headers='keys'))
    else:
        print(tabulate(dataframe.T.fillna("---"), headers='keys'))


def find_model_in_dir_or_path(dp: str):
    if os.path.isdir(dp):
        for file in os.listdir(dp):
            if file.endswith(".pt"):
                return os.path.join(dp, file)
        raise FileNotFoundError(f"There is no model file in given directory: {dp}")
    elif os.path.isfile(dp):
        if dp.endswith(".pt"):
            return dp
        raise TypeError(f"Model file must be pytorch model: {dp}")


def find_last_model_in_tree(model_trains_tree_dir):
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


def create_new_model_trains_dir(model_trains_tree_dir):
    day_dir = os.path.join(model_trains_tree_dir, datetime.now().strftime(DATE_FORMAT))
    os.makedirs(day_dir, exist_ok=True)
    max_num = 0
    for name in os.listdir(day_dir):
        _, num = name.split("_")
        max_num = max(int(num), max_num)

    dir = os.path.join(day_dir, RES_PREFIX + "_" + str(max_num + 1))
    os.makedirs(dir, exist_ok=True)

    return dir, os.path.join(dir, MODEL_NAME)


def get_model_params_count(model):
    return sum(p.numel() for p in model.parameters())


def save_history_plot(history_table, index, title, x_label, y_label, path):
    history_dict = history_table.to_dict(orient='list')
    history_dict[index] = history_table.index.tolist()
    history_dict = {key: np.array(value) for key, value in history_dict.items()}

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    x = history_dict[index]

    for key, val in history_dict.items():
        temp_x = x
        if key == 'global_epoch':
            continue

        if np.isnan(val).any():
            non_nan_mask = ~np.isnan(val)
            temp_x = x[non_nan_mask]
            val = val[non_nan_mask]
        ax.plot(temp_x, val, label=key)
        ax.legend()

    fig.savefig(path)
    plt.close(fig)
