import torch
import torchaudio
from torch.utils.data import random_split, DataLoader


def get_train_val_dataloaders(dataset, train_ratio, batch_size, val_batch_size, num_workers, val_num_workers, generator):
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator)

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
        return self.converter(waveform).transpose(-1, -2)
