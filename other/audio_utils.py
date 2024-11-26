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
import textgrids
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
        # def init(self, openslr_path, labels_path, blacklist_names=[], blacklist_readers=[]):
        self.openslr_path = openslr_path
        self.labels_path = labels_path

        self.labels = pd.read_csv(labels_path)

        args = os.path.basename(self.labels_path).split("_")
        self.sample_rate = int(args[0])
        self.vad_window = int(args[1])
        self.vad_overlap_percent = int(args[2]) / 100.0
        self.label_region_sec = int(args[3]) / 1000.0
        self.label_overlap_percent = int(args[4]) / 100.0
        self.decision_function_name = args[5]
        self.label_window = int(self.sample_rate * self.label_region_sec)
        self.label_hop = int(self.label_window * (1 - self.label_overlap_percent))

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

        return au, self.labels.labels[idx]


class MSDWildDataset(Dataset):
    def __init__(self, wild_path, window, overlap_percent, load_here=False):
        # def init(self, openslr_path, labels_path, blacklist_names=[], blacklist_readers=[]):
        self.wild_path = wild_path
        self.labels_path = os.path.join(wild_path, "rttm_label/all.rttm")
        self.wavs_path = os.path.join(wild_path, "raw_wav")

        self.sample_rate = 16000
        self.vad_window = window
        self.vad_overlap_percent = overlap_percent
        self.label_region_sec = window
        self.label_overlap_percent = overlap_percent
        # self.decision_function_name = args[5]
        self.label_window = int(self.sample_rate * self.label_region_sec)
        self.label_hop = int(self.label_window * (1 - self.label_overlap_percent))

        self._loaded = False
        self.labels = []
        if load_here:
            self.load()

    def load(self):
        with open(self.labels_path) as file:
            content = file.read()

        self._loaded = True

    def __len__(self):
        return len(os.listdir(self.wavs_path))

    def __getitem__(self, idx) -> AudioWorker:
        filename = self.labels.filename[idx]
        reader, chapter, _ = filename.split('-')
        audio_file_path = os.path.join(self.openslr_path, reader, chapter, filename)

        # name = os.path.splitext(audio_file_path)[0].replace("\\", "-")
        # au = AudioWorker(os.path.join(self.openslr_path, audio_file_path), name)
        au = AudioWorker(audio_file_path, os.path.basename(filename))
        au.load()

        return au, self.labels.labels[idx]


class EnotDataset:

    def __init__(self, pack_path, openslr_dataset):
        self.audio_path = os.path.join(pack_path, "audio")
        self.label_path = os.path.join(pack_path, "annotations")

        self.txt_files = [p for p in OpenSLRDataset.get_files_by_extension(self.label_path, ext='TextGrid')
                          if os.path.splitext(os.path.basename(p))[0] not in self.blacklist]

        self.sample_rate = openslr_dataset.sample_rate
        self.vad_window = openslr_dataset.vad_window
        self.vad_overlap_percent = openslr_dataset.vad_overlap_percent
        self.label_region_sec = openslr_dataset.label_region_sec
        self.label_overlap_percent = openslr_dataset.label_overlap_percent
        self.decision_function_name = openslr_dataset.decision_function_name
        self.label_window = openslr_dataset.label_window
        self.label_hop = openslr_dataset.label_hop

    def __getitem__(self, idx):

        audio_file_path = OpenSLRDataset.change_file_extension(self.txt_files[idx], ".wav")
        name = os.path.splitext(audio_file_path)[0]
        au = AudioWorker(os.path.join(self.audio_path, audio_file_path), name)
        au.load()
        au.resample(self.sample_rate)

        file_path = os.path.join(self.label_path, self.txt_files[idx])

        label_grid = textgrids.TextGrid(file_path)
        sample_labels, region_labels = self.convert_textgrid(label_grid, au.wave.size(-1))

        return au, sample_labels, region_labels

    @staticmethod
    def max_count_deciding(items) -> bool:
        counts = np.bincount(items)
        return bool(np.argmax(counts))

    def convert_textgrid(self, grid, sample_count):

        labeled_samples = np.zeros(sample_count, dtype='int64')

        for interval in grid['silences']:
            label = int(interval.text)
            if label == 0:
                continue

            start = int(interval.xmin * self.sample_rate)
            end = int(interval.xmax * self.sample_rate)
            if end > len(labeled_samples):
                end = len(labeled_samples)
            labeled_samples[start:end] = 1

        count = int(np.floor((len(labeled_samples) - self.label_window) / self.label_hop) + 1)
        region_labels = []
        for i in range(count):
            start = i * self.label_hop
            end = min((i + 1) * self.label_hop, len(labeled_samples))
            part = labeled_samples[start:end]
            reg_is_speech = self.max_count_deciding(part)
            region_labels.append(reg_is_speech)

        return labeled_samples, region_labels


def get_files_by_extension(directory, ext='txt', rel=False):
    pattern = os.path.join(directory, '**', f'*.{ext}')
    files = glob.glob(pattern, recursive=True)
    if rel:
        return [os.path.relpath(path, directory) for path in files]
    return files


def change_file_extension(file_path, new_extension):
    ext = new_extension.strip('.')
    return os.path.splitext(file_path)[0] + "." + ext


def parse_rttm(file_path):
    columns = ['Type', 'ID', 'Channel', 'Start', 'Duration', 'End', 'NA', 'NA_2', 'Speaker']
    df = pd.read_csv(file_path, sep=" ", names=columns, comment=';', engine='python')
    df = df[df['Type'] == 'SPEAKER']  # Filter for speaker segments
    df['Start'] = df['Start'].astype(float)
    df['End'] = df['End'].astype(float)
    return df


# Function to create binary sequence based on window and hop length
def rttm_to_binary(df, window_length, hop_length, total_duration, p):
    # Create an array with time bins based on window and hop lengths
    num_windows = int(np.floor((total_duration - window_length) / hop_length) + 1)
    binary_sequence = np.full(num_windows, False)

    # Loop through each window to check for speaker activity
    for win_idx in range(num_windows):
        # Calculate the start and end time for the window
        window_start = win_idx * hop_length
        window_end = window_start + window_length

        # Calculate total time speaker is active in this window
        active_time = 0.0

        for _, row in df.iterrows():
            # If speaker's segment overlaps with the window, calculate overlap
            if row['End'] > window_start and row['Start'] < window_end:
                overlap_start = max(row['Start'], window_start)
                overlap_end = min(row['End'], window_end)
                active_time += max(0, overlap_end - overlap_start)

        # If the active time is greater than p times the window length, mark as active (1)
        binary_sequence[win_idx] = active_time / window_length >= p

    return binary_sequence
