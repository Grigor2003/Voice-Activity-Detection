import os
from collections import defaultdict, deque
import random

import torch
from torch.utils.data import Dataset, Sampler

from other.data.audio_utils import AudioWorker, get_wav_info


class CommonAccent(Dataset):

    def __init__(self, common_accent_dir: str = None, clip_length_s=4, subset=False, datapoints: defaultdict[list] = None):
        self.sample_rate = None
        self.datapoints = defaultdict(list)
        self.clip_length_s = clip_length_s
        self.unique_labels = []

        if subset:
            self.datapoints = datapoints
            self.unique_labels = list(datapoints.keys())

        else:
            folders = os.listdir(common_accent_dir)
            folders.remove('sample_rate.txt')
            sample_rate_path = os.path.join(common_accent_dir, 'sample_rate.txt')
            with open(sample_rate_path, 'r') as f:
                self.sample_rate = int(f.readline().strip())
            self.clip_sample_count = clip_length_s * self.sample_rate

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
                    self.datapoints[label].extend((audio_path, stamp) for stamp in clip_stamps)

        self.unique_labels = sorted(self.unique_labels)
        self.label2idx = {label: i for i, label in enumerate(self.unique_labels)}
        self.idx2label = {i: label for label, i in self.label2idx.items()}
        self.label2tensor = torch.eye(len(self.unique_labels))

    def get_subset(self, subset):

        sub_dataset = CommonAccent(
            clip_length_s=self.clip_length_s,
            subset=True,
            datapoints=subset
        )

        sub_dataset.sample_rate = self.sample_rate
        sub_dataset.clip_sample_count = self.clip_sample_count
        return sub_dataset

    def get_train_val_subsets(self, train_ratio, generator):
        train_subset = defaultdict(list)
        val_subset = defaultdict(list)

        for label, samples in self.datapoints.items():
            sample_count = len(samples)
            train_sample_count = int(sample_count * train_ratio)
            inds = torch.randperm(sample_count, generator=generator)
            train_subset[label] = [samples[i] for i in inds[:train_sample_count]]
            val_subset[label] = [samples[i] for i in inds[train_sample_count:]]

        return self.get_subset(train_subset), self.get_subset(val_subset)

    @staticmethod
    def get_train_val_subsets_static(common_accent_dir, train_ratio, generator, clip_length_s=4):
        main_dataset = CommonAccent(common_accent_dir=common_accent_dir, clip_length_s=clip_length_s)

        train_dataset, val_dataset = main_dataset.get_train_val_subsets(train_ratio, generator)

        del main_dataset

        return train_dataset, val_dataset

    def get_clip_stamps(self, audio_path):
        audio_len, sr = get_wav_info(audio_path)
        if sr != self.sample_rate:
            raise ValueError('Sample rate misamatch between audio sample rate and data sample rate')
        sample_count = int(audio_len * self.sample_rate)

        clip_count = sample_count // self.clip_sample_count
        non_full_clip = sample_count % self.clip_sample_count != 0
        clip_count += non_full_clip

        clip_stamps = [i * self.clip_sample_count
                       for i in range(clip_count)]
        if sample_count - clip_stamps[-1] < self.sample_rate * 1:
            clip_stamps.pop(-1)
        return clip_stamps

    def __len__(self):
        return sum(len(samples) for samples in self.datapoints.values())

    def __getitem__(self, item) -> tuple[AudioWorker, torch.Tensor]:
        label, idx = item
        audio_path, stamp = self.datapoints[label][idx]
        label_idx = self.label2idx[label]
        label_tensor = self.label2tensor[label_idx]

        au = AudioWorker(audio_path, os.path.basename(audio_path),
                         frame_offset=stamp, num_frames=self.clip_sample_count)
        au.load()

        return au, label_tensor


class AccentSampler(Sampler):

    def __init__(self, datapoints, reset=True, shuffle=True):
        self.reset = reset  # True for train, False for validation
        self.must_stop = False
        self.shuffle = shuffle
        self.labels = list(datapoints.keys())
        self.set_lengths = {key: len(value) for key, value in datapoints.items()}
        self.longest_set = max(self.set_lengths, key=self.set_lengths.get)
        self.dataqueues = defaultdict(deque)

    def create_queue(self, label):
        self.dataqueues[label] = deque(range(self.set_lengths[label]))
        if self.shuffle:
            random.shuffle(self.dataqueues[label])

    def prepare_sampler(self):
        self.must_stop = False
        for label in self.set_lengths:
            self.create_queue(label)

    def __len__(self):
        if self.reset:
            return len(self.set_lengths.keys()) * max(self.set_lengths.values())
        return sum(self.set_lengths.values())

    def __iter__(self):
        curr_labels = list(self.labels)
        self.prepare_sampler()
        while True:
            if len(curr_labels) == 0:
                break
            label = random.choice(curr_labels)

            if len(self.dataqueues[label]) == 0:
                if not self.reset:
                    curr_labels.remove(label)
                    continue

                if label == self.longest_set:
                    self.must_stop = True
                self.create_queue(label)

            yield (label, self.dataqueues[label].pop())
