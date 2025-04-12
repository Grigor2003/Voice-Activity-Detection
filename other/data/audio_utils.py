import matplotlib.pyplot as plt
import torch
import torchaudio
import torchaudio.functional as tf
from IPython.display import Audio as PyAudio, display


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


def get_wav_frames_count(path, for_samplerate=8000):
    info = torchaudio.info(path)
    num_frames = info.num_frames
    sample_rate = info.sample_rate
    duration = for_samplerate * num_frames / sample_rate
    return duration


class AudioWorker:
    @staticmethod
    def from_wave(wave, sample_rate):
        aw = AudioWorker(None, "from wave")
        aw.rate = sample_rate
        aw.wave = wave
        aw.loaded = True
        return aw

    @property
    def wave(self):
        return self._wave

    @wave.setter
    def wave(self, value):
        self.length = value.size(1)
        self.duration_s = self.length / self.rate
        self._wave = value

    def __init__(self, path, name=None, frame_offset=0, num_frames=-1):
        self.name = name
        self.path = path

        self.__unloaded__ = "unloaded"
        self.loaded = False
        self._wave = None
        self.frame_offset = frame_offset
        self.num_frames = num_frames
        self.length = None
        self.rate = None
        self.init_rate = None
        self.duration_s = None

    def load(self):
        wave, self.rate = torchaudio.load(self.path, frame_offset=self.frame_offset, num_frames=self.num_frames)
        self.init_rate = self.rate
        self.wave = wave
        self.loaded = self.length > 0
        return self

    def resample(self, to_freq):
        if not self.loaded:
            print("WARNING: attempted to resample AudioWorker before loading")
            return self.__unloaded__
        if self.rate == to_freq:
            return self

        old_rate = self.rate
        self.rate = to_freq
        self.wave = tf.resample(self.wave, old_rate, self.rate)
        return self

    def leave_one_channel(self, channel=0):
        if not self.loaded:
            print("WARNING: attempted to change channels of AudioWorker before loading")
            return self.__unloaded__
        self._wave = self._wave[channel, :].unsqueeze(0)
        return self

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

    def clone(self):
        aw = AudioWorker(self.path, self.name, self.frame_offset, self.num_frames)
        aw.loaded = self.loaded
        if aw.loaded:
            aw.rate = self.rate
            aw.init_rate = self.init_rate
            aw.duration_s = self.duration_s
            aw.length = self.length
            aw._wave = self._wave.clone()
        return aw
