from collections import Counter

import torch
from torch.nn.utils.rnn import pad_sequence

from other.data.audio_utils import AudioWorker
from other.data.augmentation_utils import generate_white_noise, augment_with_noises, augment_volume_gain
from other.data.processing import WaveToMFCCConverter2, ChebyshevType2Filter
from other.data.work_with_stamps_utils import stamps_to_binary_counts, balance_regions, binary_counts_to_windows_np
from other.parsing.train_args_parser import NoiseArgs, ImpulseArgs
from other.utils import Example


class NoiseCollate:
    def __init__(self, sample_rate, noise_args: NoiseArgs, impulses_args: ImpulseArgs,
                 mfcc_converter: WaveToMFCCConverter2):
        self.sp_filter = None

        self.sample_rate = sample_rate
        self.noise_args = noise_args
        self.zsc = noise_args.zero_count
        self.impulses_args = impulses_args

        self.mfcc_converter = mfcc_converter

    def __call__(self, batch: list[tuple[AudioWorker, list]]):
        self.add_zero_samples(batch)
        ex_ids = [0, len(batch) - 1] + torch.randint(1, len(batch) - 1, (3,)).tolist() if len(batch) > 2 else []

        waves, targets = [], []
        examples, clear = [], None
        window = self.mfcc_converter.win_length

        for i, (aw, one_stamps) in enumerate(batch):
            binary_counts = stamps_to_binary_counts(one_stamps, aw.length)

            # Augmenting audio and balancing zero and one counts in labels
            aw.wave, binary_counts = balance_regions(aw.wave, binary_counts)

            one_counts = binary_counts_to_windows_np(binary_counts, window, aw.length)
            labels = one_counts > (window // 2)
            tar = torch.tensor(labels).float()

            if i in ex_ids:
                clear = aw.wave.clone()

            # Augmenting audio by adding real noise and white noise
            gain_augment_info = augment_volume_gain(aw)

            noise_aug_info = self.augment_aw_with_noises(aw)

            aw.wave += generate_white_noise(1, aw.length, -75 + 20 * torch.randn(1).item(), 5)

            waves.append(aw.wave.squeeze(0))
            targets.append(tar)

            if i in ex_ids:
                name = f"snr_{noise_aug_info['snrs']}"
                name = "zero_" + name if i >= len(batch) - self.zsc else name
                examples.append(Example(clear=clear, name=name,
                                        info_dicts=[noise_aug_info, gain_augment_info], i=i, label=labels))

        pad_waves = pad_sequence(waves, batch_first=True)

        use_mic_filter = torch.rand(1).item() < self.impulses_args.mic_ir_prob
        if use_mic_filter:
            mic_ind = torch.randint(len(self.impulses_args.mic_ir_loaded), (1,)).item()
            pad_inputs, exam_waves = self.mfcc_converter(pad_waves, self.impulses_args.mic_ir_loaded[mic_ind].wave,
                                                         self.sp_filter,
                                                         wave_indexes_to_return=ex_ids)
        else:
            pad_inputs, exam_waves = self.mfcc_converter(pad_waves, wave_indexes_to_return=ex_ids)

        for ex in examples:
            ex.update(wave=exam_waves[ex.i][:, :waves[ex.i].size(-1)])

        return create_batch_tensor(pad_inputs, targets), examples

    def add_zero_samples(self, batch):
        if self.zsc > 0:
            sizes = [(i.wave.size(-1), len(t), i.rate) for i, t in batch]

            inds = torch.randint(0, len(sizes), (self.zsc,))
            for i in inds:
                size, t_size, sr = sizes[i]
                aw = AudioWorker.from_wave(generate_white_noise(1, size, -50, 5), sr)
                batch.append((aw, []))

    def augment_aw_with_noises(self, aw):
        if self.noise_args.use_weights_as_counts:
            noise_datas_inds_to_counts = {j: d.weight for j, d in enumerate(self.noise_args.datas)}
        else:
            weights = torch.tensor([d.weight for d in self.noise_args.datas], dtype=torch.float)
            noise_datas_inds = torch.multinomial(weights, replacement=True, num_samples=self.noise_args.count).tolist()
            noise_datas_inds_to_counts = Counter(noise_datas_inds)

        noise_aug_info = {"snrs": []}
        for data_i, count in noise_datas_inds_to_counts.items():
            data = self.noise_args.datas[data_i]
            pool = data.loaded_pool
            noises_inds = torch.randperm(len(pool))[:count]
            snr_db = data.snr_dbs[torch.multinomial(data.snr_dbs_freqs, 1)]
            if snr_db is not None:
                noise_aug_info[data.name] = augment_with_noises(aw, [pool[j] for j in noises_inds],
                                                                data.duration_range, snr_db, data.random_phase)
            noise_aug_info["snrs"].append(snr_db)

        return noise_aug_info


class ValCollate:
    def __init__(self, sample_rate, noise_args: NoiseArgs, snr_dbs, mfcc_converter: WaveToMFCCConverter2):
        self.sample_rate = sample_rate
        self.noise_args = noise_args
        self.snr_dbs = snr_dbs
        self.mfcc_converter = mfcc_converter

    def __call__(self, batch):
        all_inputs = {snr_db: [] for snr_db in self.snr_dbs}
        all_targets = {snr_db: [] for snr_db in self.snr_dbs}

        for aw, one_stamps in batch:
            window = self.mfcc_converter.win_length
            binary_counts = stamps_to_binary_counts(one_stamps, aw.length)
            one_counts = binary_counts_to_windows_np(binary_counts, window, aw.length)
            labels = one_counts > (window // 2)
            tar = torch.tensor(labels).float()

            if self.noise_args.use_weights_as_counts:
                noise_datas_inds_to_counts = {j: d.weight for j, d in enumerate(self.noise_args.datas)}
            else:
                weights = torch.tensor([d.weight for d in self.noise_args.datas], dtype=torch.float)
                noise_datas_inds = torch.multinomial(weights, replacement=True,
                                                     num_samples=self.noise_args.count).tolist()
                noise_datas_inds_to_counts = Counter(noise_datas_inds)

            for data_i, count in noise_datas_inds_to_counts.items():
                data = self.noise_args.datas[data_i]
                pool = data.loaded_pool
                noises_inds = torch.randperm(len(pool))[:count]
                noises = [pool[j] for j in noises_inds]
                current_snr_seed = torch.randint(2 ** 32, (1,)).item()
                for snr_db in self.snr_dbs:
                    _aw = aw.clone()
                    if snr_db is not None:
                        _ = augment_with_noises(_aw, noises, data.duration_range, snr_db, data.random_phase,
                                                seed=current_snr_seed)

                    all_inputs[snr_db].append(_aw.wave[0])
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
