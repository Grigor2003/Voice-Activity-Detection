import torch
from torch.nn.utils.rnn import pad_sequence

from other.data.audio_utils import AudioWorker
from other.data.augmentation_utils import generate_white_noise, augment_with_noises, augment_volume_gain
from other.data.processing import WaveToMFCCConverter2, ChebyshevType2Filter
from other.data.work_with_stamps_utils import stamps_to_binary_counts, balance_regions, binary_counts_to_windows_np
from other.utils import Example


class NoiseCollate:
    def __init__(self, sample_rate, aug_params, snr_dbs_dict, mfcc_converter: WaveToMFCCConverter2, sp_filter,
                 zero_sample_count=0):
        self.sample_rate = sample_rate

        self.noises = None
        self.mic_irs = None
        self.sp_filter = sp_filter

        self.aug_params = aug_params
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

        ex_ids = [0, len(batch) - 1] + torch.randint(1, len(batch) - 1, (3,)).tolist() if len(batch) > 2 else []

        waves, targets = [], []
        examples, clear = [], None
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

            if i in ex_ids:
                clear = aw.wave.clone()

            # Augmenting audio by adding real noise and white noise
            gain_augment_info = augment_volume_gain(aw)
            noise_augment_info = augment_with_noises(aw, self.noises, snr_db=snr_db, **self.aug_params)
            aw.wave += generate_white_noise(1, aw.length, -75 + 20 * torch.randn(1).item(), 5)

            waves.append(aw.wave.squeeze(0))
            targets.append(tar)

            if i in ex_ids:
                name = f"snr_{snr_db}" if i < len(batch) - self.zsc else f"zero_snr_{snr_db}"
                examples.append(Example(clear=clear,
                                        name=name, info_dicts=[noise_augment_info, gain_augment_info], i=i,
                                        label=labels))

        pad_waves = pad_sequence(waves, batch_first=True)

        use_mic_filter = torch.rand(1).item() < self.aug_params["impulse_mic_prob"]
        if use_mic_filter:
            mic_ind = torch.randint(0, len(self.mic_irs) - 1, (1,)).item()
            pad_inputs, exam_waves = self.mfcc_converter(pad_waves, self.mic_irs[mic_ind].wave, self.sp_filter,
                                                         wave_indexes_to_return=ex_ids)
        else:
            pad_inputs, exam_waves = self.mfcc_converter(pad_waves, wave_indexes_to_return=ex_ids)

        for ex in examples:
            ex.update(wave=exam_waves[ex.i][:, :waves[ex.i].size(-1)])

        return create_batch_tensor(pad_inputs, targets), examples


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
                all_inputs[snr_db].append(aw.wave.squeeze(0))
                all_targets[snr_db].append(tar)
        all_inputs = {k: self.mfcc_converter(pad_sequence(v, batch_first=True)) for k, v in all_inputs.items()}
        return {snr_db: create_batch_tensor(all_inputs[snr_db], all_targets[snr_db]) for snr_db in self.snr_dbs}


def create_batch_tensor(inputs, targets):
    lengths = [t.size(0) for t in targets]
    max_len = max(lengths)
    mask = torch.arange(max_len).expand(len(lengths), max_len) < torch.tensor(lengths).unsqueeze(1)
    padded_input = pad_sequence(inputs, batch_first=True)
    padded_output = pad_sequence(targets, batch_first=True)

    return padded_input, padded_output, mask
