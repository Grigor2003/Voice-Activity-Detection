import os
from collections import Counter

import torch
from torch.nn.utils.rnn import pad_sequence

from other.data.audio_utils import AudioWorker, get_wav_frames_count
from other.data.augmentation_utils import generate_white_noise, augment_with_noises, augment_volume_gain
from other.data.processing import WaveToMFCCConverter2
from other.data.stamps_utils import balance_regions, binary_counts_to_windows_np, AudioBinaryLabel
from other.parsing.train_args_helper import NoiseArgs, ImpulseArgs
from other.utils import Example


class NoiseCollate:
    def __init__(self, sample_rate,
                 noise_args: NoiseArgs, impulses_args: ImpulseArgs,
                 mfcc_converter: WaveToMFCCConverter2, n_examples=None):
        self.spectre_filter = None

        self.sample_rate = sample_rate
        self.noise_args = noise_args
        self.impulses_args = impulses_args
        self.n_examples = n_examples

        self.mfcc_converter = mfcc_converter

    def __call__(self, batch: list[tuple[AudioWorker, AudioBinaryLabel]]):

        batch = [x for x in batch if x is not None]

        types_to_batches = {"_": batch}
        sizes = [aw.length for aw, _, _ in batch]
        # types_to_batches['zero'] = self.generate_zero_samples(sizes)

        type_to_ex_inds = {}
        if self.n_examples is not None:
            for tp, c in self.n_examples.items():
                if len(types_to_batches[tp]) > 0:
                    type_to_ex_inds[tp] = (torch.randperm(len(types_to_batches[tp]) - 1)[:c]).tolist()

        waves, targets, global_ex_inds = [], [], []
        examples, clear = [], None
        window = self.mfcc_converter.win_length

        for tp, ex_inds in type_to_ex_inds.items():
            for i, (aw, cl, abl) in enumerate(types_to_batches[tp]):
                one_counts = binary_counts_to_windows_np(abl.binary_goc(), window, aw.length)
                labels = one_counts > (window // 2)
                vad_labels = torch.tensor(labels).float()
                init_idx = torch.where(vad_labels)[0][0]
                vad_labels[init_idx:] = 1
                tar = vad_labels.unsqueeze(-1) * (cl.unsqueeze(0))
                if i in ex_inds:
                    clear = aw.wave.clone()

                # Augmenting audio by adding real noise and white noise
                gain_augment_info = augment_volume_gain(aw)

                noise_aug_info = self.augment_aw_with_noises(aw)

                aw.wave += generate_white_noise(1, aw.length, -75 + 20 * torch.randn(1).item(), 5)

                waves.append(aw.wave.squeeze(0))
                targets.append(tar)

                if i in ex_inds:
                    global_ex_inds.append(len(waves) - 1)
                    name = f"snr_{noise_aug_info['snrs']}"
                    name = tp + '_' + name
                    examples.append(Example(wave=aw.wave, clear=clear, name=name,
                                            info_dicts=[noise_aug_info, gain_augment_info], i=i, label=labels))

        pad_waves = pad_sequence(waves, batch_first=True)

        use_mic_filter = torch.rand(1).item() < self.impulses_args.mic_ir_prob
        if use_mic_filter:
            mic_ind = torch.randint(len(self.impulses_args.mic_ir_loaded), (1,)).item()
            pad_inputs, exam_waves = self.mfcc_converter(pad_waves, self.impulses_args.mic_ir_loaded[mic_ind].wave,
                                                         self.spectre_filter,
                                                         wave_indexes_to_return=global_ex_inds)
        else:
            pad_inputs, exam_waves = self.mfcc_converter(pad_waves, spectre_filter=self.spectre_filter,
                                                         wave_indexes_to_return=global_ex_inds)

        for ex, w in zip(examples, exam_waves):
            ex.update(wave=w[:, :ex.wave.size(-1)])

        return create_batch_tensor(pad_inputs, targets), examples

    def generate_zero_samples(self, sizes):
        if self.noise_args.zero_count <= 0:
            return []
        inds = torch.randint(0, len(sizes), (self.noise_args.zero_count,))
        batch = []
        for i in inds:
            aw = AudioWorker.from_wave(generate_white_noise(1, sizes[i], -50, 5), self.sample_rate)
            empty = AudioBinaryLabel.from_one_stamps([], aw.length)
            batch.append((aw, empty))
        return batch

    def augment_aw_with_noises(self, aw):
        noise_aug_info = {"snrs": []}
        if self.noise_args.count <= 0:
            return noise_aug_info

        if self.noise_args.use_weights_as_counts:
            noise_datas_inds_to_counts = {j: d.weight for j, d in enumerate(self.noise_args.datas)}
        else:
            weights = torch.tensor([d.weight for d in self.noise_args.datas], dtype=torch.float)
            noise_datas_inds = torch.multinomial(weights, replacement=True, num_samples=self.noise_args.count).tolist()
            noise_datas_inds_to_counts = Counter(noise_datas_inds)

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
    def __init__(self, sample_rate, noise_args: NoiseArgs, snr_dbs,
                 mfcc_converter: WaveToMFCCConverter2, n_examples=None):
        self.sample_rate = sample_rate
        self.noise_args = noise_args
        self.snr_dbs = snr_dbs
        self.mfcc_converter = mfcc_converter
        self.n_examples = n_examples

        self.spectre_filter = None

    def __call__(self, batch):
        all_inputs = {snr_db: [] for snr_db in self.snr_dbs}
        all_targets = {snr_db: [] for snr_db in self.snr_dbs}

        ex_inds = []
        if self.n_examples is not None:
            self.n_examples = min(self.n_examples, len(batch))
            if self.n_examples > 0:
                ex_inds = (torch.randperm(len(batch) - 1)[:self.n_examples]).tolist()

        examples = []
        self.orig_vad_labels = []
        for i, (aw, cl, abl) in enumerate(batch):
            window = self.mfcc_converter.win_length
            one_counts = binary_counts_to_windows_np(abl.binary_goc(), window, aw.length)
            labels = one_counts > (window // 2)
            vad_labels = torch.tensor(labels).float()
            self.orig_vad_labels.append(vad_labels)
            init_idx = torch.where(vad_labels)[0][0]
            vad_labels[init_idx:] = 1
            tar = vad_labels.unsqueeze(-1) * (cl.unsqueeze(0))

            if self.noise_args.use_weights_as_counts:
                noise_datas_inds_to_counts = {j: max(self.noise_args.val_min_noise_count, d.weight) for j, d in
                                              enumerate(self.noise_args.datas)}
            else:
                weights = torch.tensor([d.weight for d in self.noise_args.datas], dtype=torch.float)
                noise_datas_inds = torch.multinomial(weights, replacement=True,
                                                     num_samples=max(self.noise_args.val_min_noise_count,
                                                                     self.noise_args.count)).tolist()
                noise_datas_inds_to_counts = Counter(noise_datas_inds)

            for data_i, count in noise_datas_inds_to_counts.items():
                data = self.noise_args.datas[data_i]
                pool = data.loaded_pool
                noises_inds = torch.randperm(len(pool))[:count]
                noises = [pool[j] for j in noises_inds]
                current_snr_seed = torch.randint(2 ** 16, (1,)).item()
                for snr_db in self.snr_dbs:
                    _aw = aw.clone()
                    if snr_db is not None:
                        _ = augment_with_noises(_aw, noises, data.duration_range, snr_db, data.random_phase,
                                                seed=current_snr_seed)

                    all_inputs[snr_db].append(_aw.wave[0])
                    all_targets[snr_db].append(tar)

            if i in ex_inds:
                rand_snr = self.snr_dbs[torch.randint(len(self.snr_dbs), (1,)).item()]
                name = f"snr_{rand_snr}"
                examples.append(Example(wave=all_inputs[rand_snr][-1].unsqueeze(0), clear=aw.wave, name=name,
                                        i=i, label=labels, is_val=True))

        all_inputs = {k: self.mfcc_converter(pad_sequence(v, batch_first=True), spectre_filter=self.spectre_filter) for
                      k, v in all_inputs.items()}
        return {snr_db: create_batch_tensor(all_inputs[snr_db], all_targets[snr_db]) for snr_db in self.snr_dbs}, examples


def create_batch_tensor(inputs, targets):
    lengths = [t.size(0) for t in targets]
    max_len = max(lengths)
    mask = torch.arange(max_len).expand(len(lengths), max_len) < torch.tensor(lengths).unsqueeze(1)
    padded_input = pad_sequence(inputs, batch_first=True)
    padded_output = pad_sequence(targets, batch_first=True)
    actual_labels = padded_output.sum(dim=-1)
    mask = (mask * actual_labels).bool()
    return padded_input, padded_output, mask
