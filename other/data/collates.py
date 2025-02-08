import random

import torch
from torch.nn.utils.rnn import pad_sequence

from other.data.audio_utils import AudioWorker
from other.data.augmentation_utils import generate_white_noise, augment_sample
from other.data.work_with_stamps_utils import stamps_to_binary_counts, balance_regions, binary_counts_to_windows_np


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
        # Adding empty tracks with labels 0
        if self.zsc > 0:
            sizes = [(i.wave.size(-1), len(t), i.rate) for i, t in batch]

            for i in range(self.zsc):
                size, t_size, sr = random.choice(sizes)
                au = AudioWorker.from_wave(generate_white_noise(1, size, -50, 5), sr)
                batch.append((au, []))

        inputs, targets, examples = [], [], []
        ex_id = random.randint(1, len(batch) - 2) if len(batch) > 2 else None
        for i, (au, one_stamps) in enumerate(batch):
            total = au.wave.size(-1)
            window = self.mfcc_converter.win_length
            binary_counts = stamps_to_binary_counts(one_stamps, total)

            # Augmenting audio and balancing zero and one counts in labels
            au.wave, binary_counts = balance_regions(au.wave, binary_counts)
            total = au.wave.size(-1)

            one_counts = binary_counts_to_windows_np(binary_counts, window, total)
            labels = one_counts > (window // 2)
            tar = torch.tensor(labels).float()

            snr_db = random.choice(self.snr_dbs)

            # Augmenting audio by adding real noise and white noise
            augmented_wave, _ = augment_sample(au, self.noises, snr_db=snr_db, **self.params)
            augmented_wave += generate_white_noise(1, augmented_wave.size(-1), -50, 5)
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

        for au, one_stamps in batch:
            au.resample(self.sample_rate)
            total = au.wave.size(-1)
            window = self.mfcc_converter.win_length
            binary_counts = stamps_to_binary_counts(one_stamps, total)
            one_counts = binary_counts_to_windows_np(binary_counts, window, total)
            labels = one_counts > (window // 2)
            tar = torch.tensor(labels).float()

            for snr_db in self.snr_dbs:
                augmented_wave, _ = augment_sample(au, self.noises, snr_db=snr_db, **self.params)
                inp = self.mfcc_converter(augmented_wave)
                if tar.size(-1) != inp.size(-2):
                    print(f"WARNING: mismatch of target {tar.size(-1)} and input {inp.size(-2)} sizes in {au.name}")
                else:
                    all_inputs[snr_db].append(inp.squeeze(0))
                    all_targets[snr_db].append(tar)

        return {snr_db: create_batch_tensor(all_inputs[snr_db], all_targets[snr_db]) for snr_db in self.snr_dbs}


def create_batch_tensor(inputs, targets):
    lengths = [t.size(0) for t in inputs]
    max_len = max(lengths)
    mask = torch.arange(max_len).expand(len(lengths), max_len) < torch.tensor(lengths).unsqueeze(1)
    padded_input = pad_sequence(inputs, batch_first=True)
    padded_output = pad_sequence(targets, batch_first=True)

    return padded_input, mask, padded_output
