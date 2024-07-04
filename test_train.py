import os
import random
from time import time

import pandas as pd
from tqdm import tqdm
import torch

from torch.utils.data import DataLoader, random_split
from audio_utils import AudioWorker, OpenSLRDataset
from gru_model import SimpleG
from utils import NoiseCollate, ValidationCollate, WaveToMFCCConverter
from utils import find_last_model_in_tree, create_new_model_dir, get_validation_score

noise_data_path = r"data\noise-16k"
clean_audios_path = r"data\train-clean-100"
clean_labels_path = r"data\8000_30_50_100_50_max"

# blacklist = ['7067-76048-0021']
blacklist = []

# continue_last_model = True
continue_last_model = False

if __name__ == '__main__':

    train_ratio = 0.9

    do_epoches = 1
    epoch_noise_count = 500
    train_snr = 3
    params = {
        "noise_count": 2,
        "noise_duration_range": (5, 10),
        "snr_db": train_snr
    }
    val_params = params.copy()
    del val_params["snr_db"]
    val_snrs = [None, 10, 5, 0]
    threshold = 0.7

    dataset = OpenSLRDataset(clean_audios_path, clean_labels_path, blacklist)
    noise_files_paths = [os.path.join(noise_data_path, p) for p in os.listdir(noise_data_path) if p.endswith(".wav")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=2 ** 7, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=2 ** 7, shuffle=True, num_workers=4)

    input_size = 64
    hidden_dim = 48

    model = SimpleG(input_dim=input_size, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    bce = torch.nn.BCEWithLogitsLoss()
    bce_without_averaging = torch.nn.BCEWithLogitsLoss(reduction="sum")
    loss_history = []

    mfcc_converter = WaveToMFCCConverter(
        n_mfcc=input_size,
        sample_rate=dataset.sample_rate,
        win_length=dataset.label_window,
        hop_length=dataset.label_hop)

    train_dataloader.collate_fn = NoiseCollate(dataset.sample_rate, None, params, mfcc_converter)
    val_dataloader.collate_fn = ValidationCollate(dataset.sample_rate, None, val_params, val_snrs, mfcc_converter)

    train_name = type(model).__name__
    if continue_last_model:
        model_path = find_last_model_in_tree(train_name)

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_epoch = checkpoint['epoch']
        print(f"Loaded {model_path} with optimizer {checkpoint['optimizer']}")
        print(f"Continuing training from epoch {global_epoch}")

    else:
        model_path = create_new_model_dir(train_name)
        global_epoch = 0
        print(f"Created {model_path}")

    columns = [
        'global epoch',
        'train loss', 'train accuracy',
    ]

    for snr in val_snrs:
        if snr is None:
            columns.append('clear audio loss')
            columns.append('clear audio accuracy')
        else:
            columns.append(f'noised audio snr{snr} loss')
            columns.append(f'noised audio snr{snr} accuracy')

    training_history_table = pd.DataFrame(columns=columns)

    for epoch in range(1, do_epoches + 1):
        noises = [AudioWorker(p, p.replace("\\", "__")) for p in random.sample(noise_files_paths, epoch_noise_count)]
        for noise in noises:
            noise.load()
            noise.resample(dataset.sample_rate)

        train_dataloader.collate_fn.noises = noises
        val_dataloader.collate_fn.noises = noises

        running_loss = 0
        running_correct_count = 0
        running_whole_count = 0

        print("Training on", device)
        model.train()
        for batch_inputs, batch_targets in tqdm(train_dataloader, desc=f"epoch {global_epoch + epoch}({epoch})",
                                                disable=0):
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            output = model(batch_inputs)

            loss = bce(output, batch_targets)
            loss_history.append(loss.item())

            temp_count = batch_targets.numel()
            running_loss += loss.item() * temp_count
            running_whole_count += temp_count
            running_correct_count += torch.sum((output > threshold) == (batch_targets > threshold))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss /= running_whole_count
        accuracy = running_correct_count / running_whole_count

        # accuracy = 0

        row_values = {
            'global epoch': global_epoch + epoch,
            'train loss': running_loss, 'train accuracy': accuracy.item()
        }

        print(f"{'=' * 40}")
        print("Training scores")
        print(f"Loss: {running_loss:.4f}\nAccuracy: {accuracy:.4f}")
        print(f"{'=' * 40}")

        model.eval()
        print("Getting validation scores")
        val_loss, val_acc = get_validation_score(model, bce_without_averaging, threshold, val_snrs,
                                                 val_dataloader, device)

        print(f"{'=' * 40}")
        print("Validation scores")
        for snr in val_snrs:
            print(f"{'-' * 30}")
            if snr is None:
                name = "Clear audios"
            else:
                name = f"Noised audios snrDB {snr}"

            print(name)
            print(f"Loss: {val_loss[snr]:.4f}\nAccuracy: {val_acc[snr]:.4f}")

            if snr is None:
                row_values['clear audio loss'] = val_loss[snr].item()
                row_values['clear audio accuracy'] = val_acc[snr].item()
            else:
                row_values[f'noised audio snr{snr} loss'] = val_loss[snr].item()
                row_values[f'noised audio snr{snr} accuracy'] = val_acc[snr].item()
        print(f"{'=' * 40}\n")

        training_history_table.loc[len(training_history_table)] = row_values

    torch.save({
        'epoch': global_epoch + epoch,
        'model_state_dict': model.state_dict(),
        'optimizer': type(optimizer).__name__,
        'optimizer_state_dict': optimizer.state_dict()
    }, model_path)

    print(loss_history)
    print(training_history_table)
