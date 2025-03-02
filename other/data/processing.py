import torch
import torchaudio
from torch.utils.data import DataLoader

from other.data.datasets import AccentSampler, CommonAccent


def get_train_val_dataloaders(dataset: CommonAccent, train_ratio, batch_size, val_batch_size, num_workers, val_num_workers, generator):

    train_dataset, val_dataset = dataset.get_train_val_subsets(train_ratio, generator)

    del dataset

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  num_workers=num_workers, generator=generator,
                                  sampler=AccentSampler(train_dataset.datapoints, reset=True, shuffle=True))
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size,
                                num_workers=val_num_workers, generator=generator,
                                sampler=AccentSampler(val_dataset.datapoints, reset=False, shuffle=False))
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
        return self.converter(waveform).transpose(-1, -2)


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
        n_stft = self.n_fft // 2 + 1
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=self.f_max,
            n_stft=n_stft,
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
        mfcc = torch.matmul(mel_specgram.transpose(-1, -2), self.dct_matrix).transpose(-1, -2)
        return mfcc

    def __call__(self, audio: torch.Tensor, filter=None) -> torch.Tensor:

        spectrogram = self.get_spectrogram(audio)
        if filter is not None:
            spectrogram = filter(spectrogram)
        mfcc = self.get_mfcc(spectrogram)
        return mfcc.transpose(-1, -2)
