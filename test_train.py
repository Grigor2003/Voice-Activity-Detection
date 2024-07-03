import torch.nn as nn
from torch.utils.data import DataLoader
import os
import random
from tqdm import tqdm

from audio_utils import *
from datetime import datetime


class WaveToMFCCConverter:
    def __init__(self, n_mfcc, sample_rate=8000, frame_duration_in_ms=None, win_length=None, hop_length=None):
        self.n_mfcc = n_mfcc

        if frame_duration_in_ms is not None:
            sample_count = torch.tensor(sample_rate * frame_duration_in_ms / 1000, dtype=torch.int)
            win_length = torch.pow(2, torch.ceil(torch.log2(sample_count)).to(torch.int)).to(torch.int).item()
        elif win_length is None:
            return
        win_length = int(win_length)

        if hop_length is None:
            hop_length = int(win_length // 2)
        hop_length = int(hop_length)

        mfcc_params = {
            "n_mfcc": n_mfcc,
            "sample_rate": sample_rate
        }
        mel_params = {
            "n_fft": win_length,
            "win_length": win_length,
            "hop_length": hop_length,
            "center": False
        }

        self.converter = torchaudio.transforms.MFCC(**mfcc_params, melkwargs=mel_params)

    def __call__(self, waveform):
        return self.converter(waveform).transpose(-1, -2)


class NoiseCollate:
    def __init__(self, dataset_sample_rate, noises, params, mfcc_converter):
        self.dataset_sample_rate = dataset_sample_rate
        self.noises = noises
        self.params = params
        self.mfcc_converter = mfcc_converter

    def __call__(self, batch):
        inputs = []
        targets = []

        for au, label_txt in batch:
            au.resample(self.dataset_sample_rate)
            if self.params is None:
                augmented_wave, _ = augment_sample(au, self.noises)
            else:
                augmented_wave, _ = augment_sample(au, self.noises, **self.params)
            inp = self.mfcc_converter(augmented_wave)
            tar = torch.tensor([*map(float, label_txt)])
            if tar.size(-1) != inp.size(-2):
                print(tar.size(-1), inp.size(-2), au.name)
            inputs.append(inp)
            targets.append(tar)

        inp_dim_2 = max(i.size(1) for i in inputs)
        inputs_tens = torch.zeros([len(inputs), inp_dim_2, inputs[0].size(-1)])
        for i, inp in enumerate(inputs):
            inputs_tens[i, :inp.size(1), :] = inp

        tar_dim_1 = max(t.size(0) for t in targets)
        targets_tens = torch.zeros([len(targets), tar_dim_1, 1])
        for i, tar in enumerate(targets):
            targets_tens[i, :tar.size(0), 0] = tar

        return inputs_tens, targets_tens


class SimpleG(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(SimpleG, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out)
        return out


noise_data_path = r"data\noise-16k"
clean_audios_path = r"data\train-clean-100"
clean_labels_path = r"data\8000_30_50_100_50_max"

# blacklist = ['7067-76048-0021']
blacklist = []

continue_last_model = True

if __name__ == '__main__':

    train_name = "Simple_GRU"

    dataset = OpenSLRDataset(clean_audios_path, clean_labels_path, blacklist)
    noise_files_paths = [os.path.join(noise_data_path, p) for p in os.listdir(noise_data_path) if p.endswith(".wav")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = DataLoader(dataset, batch_size=2 ** 7, shuffle=True, num_workers=8)

    input_size = 64
    hidden_dim = 48

    model = SimpleG(input_dim=input_size, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    bce = torch.nn.BCEWithLogitsLoss()
    loss_history = []

    do_epoches = 4
    epoch_noise_count = 500
    params = {
        "noise_count": 2,
        "noise_duration_range": (5, 10),
        "snr_db": 3
    }

    mfcc_converter = WaveToMFCCConverter(
        n_mfcc=input_size,
        sample_rate=dataset.sample_rate,
        win_length=dataset.label_window,
        hop_length=dataset.label_hop)

    res_prefix = "res"
    max_num = 0

    train_results_by_model_name = os.path.join("train_results", train_name)

    if continue_last_model:
        res_dir = None

        if os.path.exists(train_results_by_model_name):
            date_objects = [datetime.strptime(date, "%Y-%m-%d")
                            for date in os.listdir(train_results_by_model_name)
                            if len(os.listdir(os.path.join(train_results_by_model_name, date))) != 0]
            if len(date_objects) != 0:
                day_dir = os.path.join(train_results_by_model_name, max(date_objects).strftime("%Y-%m-%d"))
                for name in os.listdir(day_dir):
                    st, num = name.split("_")
                    folder_path = os.path.join(day_dir, name)
                    if max_num <= int(num) and "model.pt" in os.listdir(folder_path):
                        max_num = int(num)
                        res_dir = folder_path

        if res_dir is None:
            raise ValueError("No model.pt found")
        else:
            model_path = os.path.join(res_dir, "model.pt")
            print(f"Loading {model_path}")

    else:

        day_dir = os.path.join(train_results_by_model_name, datetime.now().strftime("%Y-%m-%d"))
        os.makedirs(day_dir, exist_ok=True)
        for name in os.listdir(day_dir):
            st, num = name.split("_")
            folder_path = os.path.join(day_dir, name)
            max_num = max(int(num), max_num)
            if len(os.listdir(folder_path)) == 0:
                max_num -= 1
                break

        res_dir = os.path.join(day_dir, res_prefix + "_" + str(max_num + 1))
        os.makedirs(res_dir, exist_ok=True)

        model_path = os.path.join(res_dir, "model.pt")
        print(f"Created {model_path}")

    for epoch in range(do_epoches):
        noises = [AudioWorker(p, p.replace("\\", "__")) for p in random.sample(noise_files_paths, epoch_noise_count)]
        for noise in noises:
            noise.load()
            noise.resample(dataset.sample_rate)

        dataloader.collate_fn = NoiseCollate(dataset.sample_rate, noises, params, mfcc_converter)

        for batch_inputs, batch_targets in tqdm(dataloader, desc=f"epoch {epoch + 1}", disable=0):
            batch_inputs = batch_inputs.to(device)

            output = model(batch_inputs)

            batch_targets = batch_targets.to(device)

            loss = bce(output, batch_targets)
            loss_history.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pass

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.__ne__,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_history
        }, model_path)
        input('__________')

    print(loss_history)
