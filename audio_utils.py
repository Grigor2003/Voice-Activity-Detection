import copy
import glob
import os

import matplotlib.pyplot as plt
import torch
import torchaudio
import torchaudio.functional as tf
from IPython.display import Audio as PyAudio, display
from torch.utils.data import Dataset


class AudioWorker:
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

    def __init__(self, openslr_path, labels_pack_path, blacklist=[]):
        self.openslr_path = openslr_path
        self.labels_pack_path = labels_pack_path
        self.blacklist = blacklist

        self.txt_files = [p for p in self.get_files_by_extension(self.labels_pack_path)
                          if os.path.splitext(os.path.basename(p))[0] not in self.blacklist]

        args = os.path.basename(self.labels_pack_path).split("_")
        self.sample_rate = int(args[0])
        self.vad_window = int(args[1])
        self.vad_overlap_percent = int(args[2]) / 100.0
        self.label_region_sec = int(args[3]) / 1000.0
        self.label_overlap_percent = int(args[4]) / 100.0
        self.decision_function_name = args[5]
        self.label_window = self.sample_rate * self.label_region_sec
        self.label_hop = self.label_window * (1 - self.label_overlap_percent)

    def __len__(self):
        return len(self.txt_files)

    def __getitem__(self, idx) -> AudioWorker:
        file_path = os.path.join(self.labels_pack_path, self.txt_files[idx])
        with open(file_path, 'r') as file:
            labels_text = file.readline().strip()

        audio_file_path = self.change_file_extension(self.txt_files[idx], ".flac")
        name = os.path.splitext(audio_file_path)[0].replace("\\", "-")
        au = AudioWorker(os.path.join(self.openslr_path, audio_file_path), name)
        au.load()

        return au, labels_text


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
        print("nan after chnaging temp slice ", torch.sum(torch.isnan(temp)))
    return temp


def augment_sample(aw, noises=None, noise_count=1, noise_duration_range=(2, 5), snr_db=3):
    audio = aw.wave
    sample_rate = aw.rate

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
    temp = audio.clone()

    if noise_count > 0:

        noises_starts, _ = torch.sort(torch.rand(noise_count) * sec)
        noise_durations = torch.rand(noise_count) * (noise_duration_range[1] - noise_duration_range[0]) + \
                          noise_duration_range[0]

        noises_to_use = torch.randint(len(noises), (noise_count,))

        for i, noise_ind in enumerate(noises_to_use):
            temp = add_noise(temp,
                             noises[noise_ind],
                             snr_db=snr_db,
                             start=noises_starts[i],
                             end=-noise_durations[i],
                             sample_rate=sample_rate)

    augmentation_params = {"noises_starts": noises_starts,
                           "noise_durations": noise_durations,
                           "noises_to_use": noises_to_use}
    return temp, augmentation_params
