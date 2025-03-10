import random  # TODO: can be only temporary

import torch
from torch.nn.utils.rnn import pad_sequence

from other.data.audio_utils import AudioWorker
from other.data.augmentation_utils import generate_white_noise, augment_with_noises, augment_volume_gain
from other.data.processing import WaveToMFCCConverter2, ChebyshevType2Filter
from other.data.work_with_stamps_utils import stamps_to_binary_counts, balance_regions, binary_counts_to_windows_np


class NoiseCollate:
    def __init__(self, sample_rate, params, snr_dbs_dict, mfcc_converter: WaveToMFCCConverter2, sp_filter, zero_sample_count=0):
        self.sample_rate = sample_rate

        self.noises = None
        self.mic_irs = None
        self.sp_filter = sp_filter

        self.params = params
        self.snr_dbs, self.snr_dbs_freqs = [], []
        for snr, freq in snr_dbs_dict.items():
            self.snr_dbs.append(snr)
            self.snr_dbs_freqs.append(freq)
        self.snr_dbs_freqs = torch.tensor(self.snr_dbs_freqs, dtype=float)

        self.mfcc_converter = mfcc_converter
        self.zsc = zero_sample_count

    def __call__(self, batch):
        # Adding empty tracks with labels 0
        if self.zsc > 0:
            sizes = [(i.wave.size(-1), len(t), i.rate) for i, t in batch]

            inds = torch.randint(0, len(sizes), (self.zsc,))
            for i in inds:
                size, t_size, sr = sizes[i]
                aw = AudioWorker.from_wave(generate_white_noise(1, size, -50, 5), sr)
                batch.append((aw, []))

        ex_id = torch.randint(1, len(batch) - 1, (1,)).item() if len(batch) > 2 else None

        inputs, targets, examples = [], [], []
        for i, (aw, one_stamps) in enumerate(batch):
            binary_counts = stamps_to_binary_counts(one_stamps, aw.length)

            # Augmenting audio and balancing zero and one counts in labels
            aw.wave, binary_counts = balance_regions(aw.wave, binary_counts)

            window = self.mfcc_converter.win_length
            one_counts = binary_counts_to_windows_np(binary_counts, window, aw.length)
            labels = one_counts > (window // 2)
            tar = torch.tensor(labels).float()

            snr_db_ind = torch.multinomial(self.snr_dbs_freqs, 1)
            snr_db = self.snr_dbs[snr_db_ind]

            # Augmenting audio by adding real noise and white noise
            gain_augment_info = augment_volume_gain(aw)
            noise_augment_info = augment_with_noises(aw, self.noises, snr_db=snr_db, **self.params)
            aw.wave += generate_white_noise(1, aw.length, -50, 5)


            inp = self.mfcc_converter(aw.wave, random.choice(self.mic_irs).wave, self.sp_filter)

            if tar.size(-1) != inp.size(-2):
                print(f"WARNING: mismatch of target {tar.size(-1)} and input {inp.size(-2)} sizes in {aw.name}")
            else:
                inputs.append(inp.squeeze(0))
                targets.append(tar)
            if i == ex_id or i == 0 or i == len(batch) - 1:
                examples.append([i, aw.wave.clone(), f"snr{snr_db}", noise_augment_info, gain_augment_info])

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

        for aw, one_stamps in batch:
            aw.resample(self.sample_rate)
            total = aw.wave.size(-1)
            window = self.mfcc_converter.win_length
            binary_counts = stamps_to_binary_counts(one_stamps, total)
            one_counts = binary_counts_to_windows_np(binary_counts, window, total)
            labels = one_counts > (window // 2)
            tar = torch.tensor(labels).float()

            for snr_db in self.snr_dbs:
                noise_augment_info = augment_with_noises(aw, self.noises, snr_db=snr_db, **self.params)
                inp = self.mfcc_converter(aw.wave)
                if tar.size(-1) != inp.size(-2):
                    print(f"WARNING: mismatch of target {tar.size(-1)} and input {inp.size(-2)} sizes in {aw.name}")
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
