import numpy as np
import scipy.signal as signal
import torch
import torchaudio
from torch.utils.data import random_split, DataLoader


def get_train_val_dataloaders(dataset, train_ratio, batch_size, val_batch_size, num_workers, val_num_workers,
                              generator):
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                              torch.Generator().manual_seed(generator.initial_seed()))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers, generator=generator)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size,
                                shuffle=True, num_workers=val_num_workers, generator=generator)
    return train_dataloader, val_dataloader


class WaveToMFCCConverter:
    def __init__(self, n_mfcc, sample_rate=8000, frame_duration_in_ms=None, win_length=None, hop_length=None):
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
        self.frame_duration_in_ms = frame_duration_in_ms

        if frame_duration_in_ms is not None:
            sample_count = torch.tensor(sample_rate * frame_duration_in_ms / 1000, dtype=torch.int)
            win_length = torch.pow(2, torch.ceil(torch.log2(sample_count)).to(torch.int)).to(torch.int).item()
        elif win_length is None:
            win_length = sample_rate // 20
        win_length = int(win_length)

        if hop_length is None:
            hop_length = int(win_length // 2)
        hop_length = int(hop_length)

        self.win_length = win_length
        self.hop_length = hop_length

        mfcc_params = {
            "n_mfcc": n_mfcc,
            "sample_rate": sample_rate,
            "log_mels": False
        }
        mel_params = {
            "n_fft": win_length,
            "win_length": win_length,
            "hop_length": hop_length,
            "center": False,
            "norm": 'slaney'
        }

        self.converter = torchaudio.transforms.MFCC(**mfcc_params, melkwargs=mel_params)

    def __call__(self, waveform):
        return self.converter(waveform).mT


class WaveToMFCCConverter2:
    def __init__(
            self,
            sample_rate: int,
            n_fft: int = None,
            hop_length: int = None,
            win_length: int = None,
            window_fn: callable = torch.hann_window,
            power: float = 2.0,
            n_mels: int = 128,
            f_min: float = 0.0,
            f_max: float = None,
            mel_norm: str = "slaney",
            mel_scale: str = "htk",
            n_mfcc: int = 20,
            log_mels: bool = False,
            dct_norm: str = "ortho",
            center: bool = False
    ):
        super().__init__()
        self.n_fft = n_fft if n_fft is not None else win_length
        self.hop_length = hop_length if hop_length is not None else n_fft // 2
        self.win_length = win_length if win_length is not None else n_fft
        self.power = power
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate // 2
        self.mel_norm = mel_norm
        self.mel_scale_type = mel_scale
        self.top_db = 80.0
        self.n_mfcc = n_mfcc
        self.log_mels = log_mels
        self.dct_norm = dct_norm

        # Spectrogram transform
        self.spectrogram_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window_fn=window_fn,
            power=power,
            center=center,
            pad_mode="reflect",
            onesided=True,
        )

        # Mel scale transform
        self.n_stft = self.n_fft // 2 + 1
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=self.f_max,
            n_stft=self.n_stft,
            norm=mel_norm,
            mel_scale=mel_scale,
        )

        self.amplitude_to_DB = torchaudio.transforms.AmplitudeToDB("power", self.top_db)

        # Create DCT matrix for MFCC
        self.dct_matrix = torchaudio.functional.create_dct(
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            norm=dct_norm,
        )

    def get_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio waveform to spectrogram.

        Args:
            audio: Input tensor of shape (sample_count,) or (1, sample_count)

        Returns:
            Spectrogram tensor of shape (1, n_fft//2 + 1, frame_count)
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        spec = self.spectrogram_transform(audio)
        return spec

    def get_mfcc(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Convert spectrogram to MFCC coefficients.

        Args:
            spectrogram: Input spectrogram tensor of shape (1, freq_bins, frame_count)

        Returns:
            MFCC tensor of shape (1, n_mfcc, frame_count)
        """
        mel_specgram = self.mel_scale(spectrogram)
        if self.log_mels:
            mel_specgram = torch.log(mel_specgram + 1e-6)
        else:
            mel_specgram = self.amplitude_to_DB(mel_specgram)

        # Apply DCT to get MFCCs
        mfcc = torch.matmul(mel_specgram.mT, self.dct_matrix).mT
        return mfcc

    def __call__(self, audio: torch.Tensor, ir=None, spectre_filter=None) -> torch.Tensor:

        if ir is not None:
            orig_len = audio.shape[-1]
            audio = torchaudio.functional.fftconvolve(audio, ir, mode='same')
            audio = audio[:, :orig_len]

        spectrogram = self.get_spectrogram(audio)

        if spectre_filter is not None:
            if isinstance(spectre_filter, list) and all(callable(item) for item in spectre_filter):
                for f in spectre_filter:
                    spectrogram = f(spectrogram)
            elif callable(spectre_filter):
                spectrogram = spectre_filter(spectrogram)
            else:
                raise TypeError("spectre_filter should either be list[callable] or callable")

        mfcc = self.get_mfcc(spectrogram)
        return mfcc.mT


class PassFilter:

    def __init__(
            self,
            sample_rate,
            n_fft,
            lower_bound=None,
            upper_bound=None,
            band=None
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_stft = n_fft // 2 + 1
        self.resolution = sample_rate // n_fft

        if lower_bound is not None:
            self.lower_bound_bin_idx = lower_bound // self.resolution
        if upper_bound is not None:
            self.upper_bound_bin_idx = upper_bound // self.resolution
        if band is not None:
            self.band_slice = slice(band[0] // self.resolution, band[1] // self.resolution + 1)

    def HPF(self, specgram):
        specgram[:, :self.lower_bound_bin_idx + 1] = 0
        return specgram

    def LPF(self, specgram):
        specgram[:, self.upper_bound_bin_idx:] = 0
        return specgram

    def BPF(self, specgram):
        temp = specgram[:, self.band_slice]
        specgram = torch.zeros_like(specgram)
        specgram[:, self.band_slice] = temp
        return specgram


class ChebyshevType2Filter:

    def __init__(
            self,
            sample_rate,
            n_fft,
            lower_bound=None,
            upper_bound=None,
            band=None
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_stft = n_fft // 2 + 1

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.band = band

        A_stop = 40  # Stopband attenuation in dB
        R_pass = 0.5  # Passband ripple in dB
        N = 9  # Filter order
        nyquist = 0.5 * sample_rate

        if lower_bound is not None:
            Wn = lower_bound / nyquist
            self.pf_tensor = self.get_filter_tensor(N, Wn, R_pass, A_stop, 'high', self.n_stft)

        if upper_bound is not None:
            Wn = upper_bound / nyquist
            self.pf_tensor = self.get_filter_tensor(N, Wn, R_pass, A_stop, 'low', self.n_stft)

        if band is not None:
            Wn = (band[0] / nyquist, band[1] / nyquist)
            self.pf_tensor = self.get_filter_tensor(N, Wn, R_pass, A_stop, 'band', self.n_stft)

    def __call__(self, spec):
        return (spec.mT * self.pf_tensor).mT

    @staticmethod
    def get_filter_tensor(N, Wn, R_pass, A_stop, btype, worN):
        x = signal.iirfilter(N, Wn, rp=R_pass, rs=A_stop, btype=btype, ftype='cheby2')
        _, h = signal.freqz(*x, worN=worN)
        return torch.tensor(np.abs(h), dtype=torch.float32)