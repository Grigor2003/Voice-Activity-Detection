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
    @staticmethod
    def get_files_by_extension(directory, ext='txt'):
        pattern = os.path.join(directory, '**', f'*.{ext}')
        return [os.path.relpath(path, directory) for path in glob.glob(pattern, recursive=True)]

    @staticmethod
    def change_file_extension(file_path, new_extension):
        ext = new_extension.strip('.')
        return os.path.splitext(file_path)[0] + "." + ext

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


def calculate_rms(tensor):
    return torch.sqrt(torch.mean(tensor ** 2))


def add_noise(audio, noise, snr_db, start, end, in_seconds=True, sample_rate=8000):
    if start < 0:
        start = 0
    if end is not None:
        if end < 0:
            end = start - end
    if in_seconds:
        start = int(start * sample_rate)
        if start >= audio.size(-1):
            return audio
        if end is None:
            end = audio.size(-1)
        else:
            end = min(audio.size(-1), int(end * sample_rate))
    if end is None:
        end = audio.size(-1)
    audio_part = audio[:, start:end]
    orig_noise = noise.clone()
    if torch.sum(torch.isnan(orig_noise)):
        print("orig noise has nan")
    while end - start > noise.size(-1):
        noise = torch.cat([noise, orig_noise], dim=1)
    noise = noise[:, : end - start]
    if torch.sum(torch.isnan(noise)):
        print("nan after repeating nosie ", torch.sum(torch.isnan(orig_noise)), torch.sum(torch.isnan(noise)))
    #     noised = tf.add_noise(audio_part, noise, torch.Tensor([snr_db]))

    signal_rms = calculate_rms(audio_part)

    # Calculate RMS of the noise
    noise_rms = calculate_rms(noise)

    # Calculate desired noise RMS based on SNR
    snr = 10 ** (snr_db / 20)
    desired_noise_rms = signal_rms / snr

    # Scale noise to match the desired RMS
    scaled_noise = noise * (desired_noise_rms / (noise_rms + 1e-10))  # Adding small value to avoid division by zero

    # Add the scaled noise to the original signal
    noised = audio_part + scaled_noise

    # Debugging: Check for NaNs or Infs
    if torch.isnan(noised).any() or torch.isinf(noised).any():
        print("Noisy waveform contains NaNs or Infs")

    if torch.sum(torch.isnan(noised)):
        print("nan after applying noise ",
              audio[:, start:end].size(),
              noise.size(),
              torch.Tensor([snr_db]),
              torch.sum(torch.isnan(noised)))
    temp = audio.clone()
    temp[:, start:end] = noised
    if torch.sum(torch.isnan(temp)):
        print("nan after changing temp slice ", torch.sum(torch.isnan(temp)))
    return temp


def augment_sample(aw, noises=None, noise_count=1, noise_duration_range=(2, 5), snr_db=3):
    if None in [noises, noise_count, noise_duration_range, snr_db]:
        return aw.wave, None

    audio = aw.wave
    sample_rate = aw.rate
    orig_audio = audio.clone()
    augmentation_params = None

    resampled_noises = []
    for noise in noises:
        if noise.rate != sample_rate:
            temp = noise.deepcopy()
            temp.resample(sample_rate)
            resampled_noises.append(temp.wave)
        else:
            resampled_noises.append(noise.wave)
    noises = resampled_noises

    sec = audio.size(-1) / sample_rate

    if noise_count <= 0:
        return orig_audio, augmentation_params

    noises_starts, _ = torch.sort(torch.rand(noise_count) * sec)
    noise_durations = torch.rand(noise_count) * (noise_duration_range[1] - noise_duration_range[0]) + \
                      noise_duration_range[0]

    noises_to_use = torch.randint(len(noises), (noise_count,))

    for i, noise_ind in enumerate(noises_to_use):
        orig_audio = add_noise(orig_audio,
                               noises[noise_ind],
                               snr_db=snr_db,
                               start=noises_starts[i],
                               end=-noise_durations[i],
                               sample_rate=sample_rate)

    augmentation_params = {"noises_starts": noises_starts,
                           "noise_durations": noise_durations,
                           "noises_to_use": noises_to_use}

    return orig_audio, augmentation_params


def generate_white_noise(count, samples_count, noise_db=1, noise_dev=0):
    dbs = torch.normal(noise_db, noise_dev, size=(count, 1))
    noise_power_linear = 10 ** (dbs / 10)
    noise = torch.randn(count, samples_count)
    noise = noise * noise_power_linear
    return noise
