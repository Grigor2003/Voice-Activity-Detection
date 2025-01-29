import copy
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchaudio
import torchaudio.functional as tf
from IPython.display import Audio as PyAudio, display
from torch.utils.data import Dataset
import numpy as np


class AudioWorker:
    @staticmethod
    def from_wave(wave, sample_rate):
        au = AudioWorker(None, "from wave")
        au.wave = wave
        au.rate = sample_rate
        au.loaded = True
        return au

    def __init__(self, path, name=None):

        self.name = name
        self.path = path

        self.__unloaded__ = "unloaded"
        self.loaded = False
        self.wave = None
        self.rate = None

    def load(self):
        self.wave, self.rate = torchaudio.load(self.path)
        self.loaded = self.wave.size(1) > 0
        return self

    def resample(self, to_freq):
        if not self.loaded:
            print("WARNING: attempted to resample AudioWorker before loading")
            return self.__unloaded__
        if self.rate == to_freq:
            return

        self.wave = tf.resample(self.wave, self.rate, to_freq)
        self.rate = to_freq

    def player(self, mask=None):
        if not self.loaded:
            return self.__unloaded__
        if mask is None:
            return display(PyAudio(self.wave, rate=self.rate, autoplay=False))
        else:
            return display(PyAudio(self.wave[:, mask], rate=self.rate, autoplay=False))

    def plot_waveform(self, mask=None, regions=None, win_length=None, hop_length=None, region_linewidth=None):
        if not self.loaded:
            return self.__unloaded__
        waveform = self.wave.numpy()

        num_channels, num_frames = waveform.shape
        time = torch.arange(0, num_frames) / self.rate

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]

        for c in range(num_channels):
            values = waveform[c]

            if mask is not None:
                axes[c].plot(time[~mask], values[~mask], "o", linewidth=1, c="red")

            axes[c].plot(time, values, linewidth=1, c="green")

            if not (None in [regions, win_length, hop_length, region_linewidth]):
                max_val = values.max()
                for i, label in enumerate(regions):
                    s = i * hop_length
                    e = s + win_length
                    plot_values = [max_val / ((label == 1) + 1)] * hop_length
                    plot_color = "green" if (label == 1) else "red"
                    axes[c].plot(time[s:e], plot_values, c=plot_color, linewidth=region_linewidth)

            axes[c].grid(True)
            if num_channels > 1:
                axes[c].set_ylabel(f"Channel {c + 1}")
        figure.suptitle(f"Waveform of {self.name}")

    def deepcopy(self):
        return copy.deepcopy(self)


class OpenSLRDataset(Dataset):
    def __init__(self, openslr_path, labels_path):
        self.openslr_path = openslr_path
        self.labels_path = labels_path

        self.labels = pd.read_csv(labels_path)

        args = os.path.basename(self.labels_path).split("_")
        self.sample_rate = int(args[0])
        self.vad_window = int(args[1])
        self.vad_overlap_percent = int(args[2]) / 100.0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> AudioWorker:
        filename = self.labels.filename[idx]
        reader, chapter, _ = filename.split('-')
        audio_file_path = os.path.join(self.openslr_path, reader, chapter, filename)

        # name = os.path.splitext(audio_file_path)[0].replace("\\", "-")
        # au = AudioWorker(os.path.join(self.openslr_path, audio_file_path), name)
        au = AudioWorker(audio_file_path, os.path.basename(filename))
        au.load()

        stamps_flatten = [*map(int, self.labels.at[idx, 'labels'].split('-'))]
        stamps = list(zip(stamps_flatten[::2], stamps_flatten[1::2]))

        return au, stamps


class MSDWildDataset(Dataset):
    def __init__(self, wavs_path, rttm_path, target_rate):
        self.wavs_path = wavs_path
        self.rttm_path = rttm_path

        self.labels_df = parse_rttm(rttm_path, target_rate)
        # args = os.path.basename(self.labels_path).split("_")
        # self.sample_rate = int(args[0])
        # self.vad_window = int(args[1])
        # self.vad_overlap_percent = int(args[2]) / 100.0

    # def __len__(self):
    #     return len(self.labels)

    # def __getitem__(self, idx) -> AudioWorker:
    #     filename = self.labels.filename[idx]
    #     reader, chapter, _ = filename.split('-')
    #     audio_file_path = os.path.join(self.openslr_path, reader, chapter, filename)
    #
    #     # name = os.path.splitext(audio_file_path)[0].replace("\\", "-")
    #     # au = AudioWorker(os.path.join(self.openslr_path, audio_file_path), name)
    #     au = AudioWorker(audio_file_path, os.path.basename(filename))
    #     au.load()
    #
    #     stamps_flatten = self.labels.at[idx, 'labels'].split('-')
    #     stamps = list(zip(stamps_flatten[::2], stamps_flatten[1::2]))
    #
    #     return au, stamps


def get_files_by_extension(directory, ext='txt', rel=False):
    pattern = os.path.join(directory, '**', f'*.{ext}')
    files = glob.glob(pattern, recursive=True)
    if rel:
        return [os.path.relpath(path, directory) for path in files]
    return files


def change_file_extension(file_path, new_extension):
    ext = new_extension.strip('.')
    return os.path.splitext(file_path)[0] + "." + ext


def parse_rttm(labels_path, sr):
    data = {}
    with open(labels_path, 'r') as f:
        for line in f:
            splitted = line.split(' ')
            if len(splitted) < 4:
                continue
            filename = splitted[1]
            time_begin = int(float(splitted[3]) * sr)
            time_duration = int(float(splitted[4]) * sr)

            file_path = filename + '.wav'
            if file_path in data.keys():
                data[file_path].append((time_begin, time_begin + time_duration))
            else:
                data[file_path] = [(time_begin, time_begin + time_duration)]

    return data
