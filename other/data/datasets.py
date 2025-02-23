import os

import torch
import pandas as pd

from torch.utils.data import Dataset
from other.data.audio_utils import AudioWorker, get_wav_info


class CommonAccent(Dataset):
    def __init__(self, common_accent_csv, clip_length=4):
        self.datapoints = []
        self.clip_length = clip_length

        df = pd.read_csv(common_accent_csv)
        self.unique_labels = sorted(df['label'].unique())
        self.label2idx = {label: i for i, label in enumerate(self.unique_labels)}
        self.idx2label = {i: label for label, i in self.label2idx.items()}
        self.label2tensor = torch.eye(len(self.unique_labels))
        for idx, row in df.iterrows():
            audio_path = row['audio_path']
            label = row['label']
            label_idx = self.label2idx[label]
            clip_stamps = self.get_clip_stamps(audio_path)
            self.datapoints.extend((audio_path, stamp, label_idx) for stamp in clip_stamps)

    def get_clip_stamps(self, audio_path):
        audio_len, sr = get_wav_info(audio_path)
        clip_sample_count = self.clip_length * sr
        sample_count = int(audio_len * sr)

        clip_count = sample_count // clip_sample_count
        non_full_clip = sample_count % clip_sample_count != 0
        clip_count += non_full_clip

        clip_stamps = [(i * clip_sample_count, (i+1) * clip_sample_count)
                       for i in range(clip_count)]
        if non_full_clip:
            clip_stamps[-1] = ((clip_count - 1) * clip_sample_count, sample_count)
        return clip_stamps

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx) -> AudioWorker:

        audio_path, stamp, label_idx = self.datapoints[idx]
        label_tensor = self.label2tensor[label_idx]

        au = AudioWorker(audio_path, os.path.basename(audio_path), frame_offset=stamp[0], num_frames=stamp[1])
        au.load()

        return au, label_tensor
