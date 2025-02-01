from torch.utils.data import Dataset
import pandas as pd
import os

from other.data.audio_utils import AudioWorker, parse_rttm


class OpenSLRDataset(Dataset):
    def __init__(self, openslr_path, labels_path):
        self.openslr_path = openslr_path
        self.labels_path = labels_path

        self.labels = pd.read_csv(labels_path)

        args = os.path.basename(self.labels_path).split("_")
        self.sample_rate = int(args[0])
        self.vad_window = int(args[1])
        self.vad_overlap_percent = int(args[2]) / 100.0

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

        stamps_flatten = [*map(int, self.labels.at[idx, 'labels'].split('-'))]
        stamps = list(zip(stamps_flatten[::2], stamps_flatten[1::2]))

        return au, stamps


class MSDWildDataset(Dataset):
    def __init__(self, wavs_path, rttm_path, target_rate):
        self.wavs_path = wavs_path
        self.rttm_path = rttm_path

        self.labels_df = parse_rttm(rttm_path, target_rate)
        # args = os.path.basename(self.labels_path).split("_")
        # self.sample_rate = int(args[0])
        # self.vad_window = int(args[1])
        # self.vad_overlap_percent = int(args[2]) / 100.0

    # def __len__(self):
    #     return len(self.labels)

    # def __getitem__(self, idx) -> AudioWorker:
    #     filename = self.labels.filename[idx]
    #     reader, chapter, _ = filename.split('-')
    #     audio_file_path = os.path.join(self.openslr_path, reader, chapter, filename)
    #
    #     # name = os.path.splitext(audio_file_path)[0].replace("\\", "-")
    #     # au = AudioWorker(os.path.join(self.openslr_path, audio_file_path), name)
    #     au = AudioWorker(audio_file_path, os.path.basename(filename))
    #     au.load()
    #
    #     stamps_flatten = self.labels.at[idx, 'labels'].split('-')
    #     stamps = list(zip(stamps_flatten[::2], stamps_flatten[1::2]))
    #
    #     return au, stamps
