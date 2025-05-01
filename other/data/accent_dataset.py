import os
from collections import defaultdict, deque
import random

import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from other.data.audio_utils import AudioWorker
from other.data.stamps_utils import AudioBinaryLabel


class CommonAccent(Dataset):

    def __init__(self, common_accent_dir: str = None, vad_labels_path: str = None, subset=False, datapoints: defaultdict[list] = None):
        self.sample_rate = None
        self.datapoints = defaultdict(list)
        self.unique_labels = []
        self.vad_labels = None

        if vad_labels_path:
            df = pd.read_csv(vad_labels_path)
            self.vad_labels = dict(zip(df['filename'], df['labels']))

        if subset:
            self.datapoints = datapoints
            self.unique_labels = list(datapoints.keys())

        else:
            folders = os.listdir(common_accent_dir)
            folders.remove('sample_rate.txt')
            sample_rate_path = os.path.join(common_accent_dir, 'sample_rate.txt')
            with open(sample_rate_path, 'r') as f:
                self.sample_rate = int(f.readline().strip())

            for folder in folders:
                folder_path = os.path.join(common_accent_dir, folder)
                files = os.listdir(folder_path)
                files.remove('accent.txt')
                # files = files[:100]
                label_path = os.path.join(folder_path, 'accent.txt')
                with open(label_path, 'r') as f:
                    label = f.readline().strip()
                self.unique_labels.append(label)
                self.datapoints[label].extend((os.path.join(folder_path, file),
                                              os.path.join(folder, file)) for file in files)

        self.unique_labels = sorted(self.unique_labels)
        self.label2idx = {label: i for i, label in enumerate(self.unique_labels)}
        self.idx2label = {i: label for label, i in self.label2idx.items()}
        self.idx2tensor = torch.eye(len(self.unique_labels))

    def label2tensor(self, label):
        return self.idx2tensor[self.label2idx[label]]

    def get_subset(self, subset):

        sub_dataset = CommonAccent(
            subset=True,
            datapoints=subset
        )

        sub_dataset.sample_rate = self.sample_rate
        sub_dataset.vad_labels = self.vad_labels
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

    def __len__(self):
        return sum(len(samples) for samples in self.datapoints.values())

    def __getitem__(self, item) -> tuple[AudioWorker, torch.Tensor]:
        label, idx = item
        audio_path, row_name = self.datapoints[label][idx]
        label_tensor = self.label2tensor(label)
        vad_stamps = list(map(int, self.vad_labels[row_name].split('-')))
        vad_stamps = [(s, e) for s, e in zip(vad_stamps[::2], vad_stamps[1::2])]
        au = AudioWorker(audio_path, os.path.basename(audio_path))
        au.load()
        abl = AudioBinaryLabel.from_one_stamps(vad_stamps, length=au.length)

        return au, label_tensor, abl


class AccentSampler(Sampler):

    def __init__(self, datapoints, reset=True, shuffle=True):
        self.reset = reset  # True for train, False for validation
        self.must_stop = False
        self.shuffle = shuffle
        self.labels = list(datapoints.keys())
        self.set_lengths = {key: len(value) for key, value in datapoints.items()}
        self.longest_set = max(self.set_lengths, key=self.set_lengths.get)
        self.data_queues = defaultdict(deque)

    def create_queue(self, label):
        self.data_queues[label] = deque(range(self.set_lengths[label]))
        if self.shuffle:
            random.shuffle(self.data_queues[label])

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

            if len(self.data_queues[label]) == 0:
                if not self.reset:
                    curr_labels.remove(label)
                    continue

                if label == self.longest_set:
                    self.must_stop = True
                self.create_queue(label)

            yield label, self.data_queues[label].pop()
