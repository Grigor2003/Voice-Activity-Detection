import os

import torch
import pandas as pd

from torch.utils.data import Dataset
from other.data.audio_utils import AudioWorker, get_wav_info


class CommonAccent(Dataset):
    def __init__(self, common_accent_dir, clip_length_s=4):
        self.sample_rate = None
        self.datapoints = []
        self.clip_length_s = clip_length_s
        self.unique_labels = []

        folders = os.listdir(common_accent_dir)
        folders.remove('sample_rate.txt')
        sample_rate_path = os.path.join(common_accent_dir, 'sample_rate.txt')
        with open(sample_rate_path, 'r') as f:
            self.sample_rate = int(f.readline().strip())

        for folder in folders:
            folder_path = os.path.join(common_accent_dir, folder)
            files = os.listdir(folder_path)
            files.remove('accent.txt')
            label_path = os.path.join(folder_path, 'accent.txt')
            with open(label_path, 'r') as f:
                label = f.readline().strip()
            self.unique_labels.append(label)

            for file in files:
                audio_path = os.path.join(folder_path, file)
                clip_stamps = self.get_clip_stamps(audio_path)
                self.datapoints.extend((audio_path, stamp, label) for stamp in clip_stamps)

        self.unique_labels = sorted(self.unique_labels)
        self.label2idx = {label: i for i, label in enumerate(self.unique_labels)}
        self.idx2label = {i: label for label, i in self.label2idx.items()}
        self.label2tensor = torch.eye(len(self.unique_labels))

    def get_clip_stamps(self, audio_path):
        audio_len, sr = get_wav_info(audio_path)
        clip_sample_count = self.clip_length_s * sr
        sample_count = int(audio_len * sr)

        clip_count = sample_count // clip_sample_count
        non_full_clip = sample_count % clip_sample_count != 0
        clip_count += non_full_clip

        clip_stamps = [(i * clip_sample_count, clip_sample_count)
                       for i in range(clip_count)]
        if sample_count - clip_stamps[-1][0] < self.sample_rate * 0.5:
            clip_stamps.pop(-1)
        return clip_stamps

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx) -> AudioWorker:

        audio_path, stamp, label = self.datapoints[idx]
        label_idx = self.label2idx[label]
        label_tensor = self.label2tensor[label_idx]

        au = AudioWorker(audio_path, os.path.basename(audio_path), frame_offset=stamp[0], num_frames=stamp[1])
        au.load()

        return au, label_tensor
