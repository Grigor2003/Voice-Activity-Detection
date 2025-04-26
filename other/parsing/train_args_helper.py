import torch

from other.parsing.parsing_utils import *


class SynthArgs:
    def __init__(self, dct):
        self.labels_path = is_type_of(dct['labels'], req=False)
        self.dir = is_type_of(dct['dir'], req=self.labels_path is not None)
        self.default_comb_count = is_range(dct['default_comb_count'], 1, 100, req=self.labels_path is not None)

        self.synth_arg = is_type_of(dct['synth_arg'], (int, float), req=self.labels_path is not None)
        self.zero_arg = is_type_of(dct['zero_arg'], (int, float))

        self.count = 0
        self.zero_count = 0

        self.labels = dict()
        self.paths = []
        self.count = 0

    def post_count(self, batch_size):
        if self.synth_arg < 0:
            self.count = int(-self.synth_arg)
        elif self.synth_arg > 0:
            self.count = int(self.synth_arg * batch_size)

    def post_zero_count(self, batch_size):
        if self.zero_arg < 0:
            self.zero_count = int(-self.zero_arg)
        elif self.zero_arg > 0:
            self.zero_count = int(self.zero_arg * batch_size)


class NoiseArgs:
    def __init__(self, dct):
        self.count = is_range(dct['count'], 0, 100, int)
        self.use_weights_as_counts = is_type_of(dct['use_weights_as_counts'], bool)
        default_snr_to_frec = parse_numeric_dict(dct['snr&weight'],
                                                 1, 100,
                                                 [-25, 25, True, False],
                                                 [0, 2 ** 16, True, True])

        self.val_min_noise_count = 1

        self.datas = []
        for name, d in dct.items():
            if not isinstance(d, dict) or name in ["snr&weight"]:
                continue
            self.datas.append(NoiseData(name, d, default_snr_to_frec))


class NoiseData:
    def __init__(self, name, dct, default_snr_dict):
        self.name = name
        self.weight = is_range(dct['weight'], 0, 5000, int)
        self.data_dir = is_type_of(dct['dir'])
        self.epoch_pool = is_range(dct['epoch_pool'], 0, 5000, int)
        self.duration_range = parse_range(dct['duration'], [0, 60], [0, 60])
        self.random_phase = is_type_of(dct['random_phase'], bool)
        if 'snr&weight' in dct.keys():
            snr_to_freq_dict = parse_numeric_dict(dct['snr&weight'],
                                                  1, 100,
                                                  [-25, 25, True, False],
                                                  [0, 2 ** 16, True, True])
        else:
            snr_to_freq_dict = default_snr_dict
        self.snr_dbs, self.snr_dbs_freqs = [], []
        for snr, freq in snr_to_freq_dict.items():
            self.snr_dbs.append(snr)
            self.snr_dbs_freqs.append(freq)
        self.snr_dbs_freqs = torch.tensor(self.snr_dbs_freqs, dtype=torch.float)

        self.all_files_paths = []
        self.loaded_pool = []


class ImpulseArgs:
    def __init__(self, dct):
        self.mic_ir_dir = is_type_of(dct['mic_ir_dir'], req=False)
        self.mic_ir_prob = is_range(dct['mic_ir_prob'], 0, 1, req=self.mic_ir_dir is not None)
        self.mic_ir_files_paths = []
        self.mic_ir_loaded = []
