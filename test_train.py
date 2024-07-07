import os
import random
import time

import pandas as pd
from tqdm import tqdm
import torch

from torch.utils.data import DataLoader, random_split
from audio_utils import AudioWorker, OpenSLRDataset
from gru_model import SimpleG
from utils import NoiseCollate, ValidationCollate, WaveToMFCCConverter
from utils import find_last_model_in_tree, create_new_model_trains_dir, get_validation_score

noise_data_path = r"data\noise-16k"
clean_audios_path = r"data\train-clean-100"
clean_labels_path = r"data\8000_30_50_100_50_max"

train_name = ""

# blacklist = ['7067-76048-0021']
blacklist = []

continue_last_model = True
# continue_last_model = False

models_root_dir = "train_results"

if __name__ == '__main__':

    train_ratio = 0.9

    do_epoches = 1
    train_num_workers = 4
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

    train_dataloader = DataLoader(train_dataset, batch_size=2 ** 7, shuffle=True, num_workers=train_num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=2 ** 7, shuffle=True, num_workers=4)

    input_size = 64
    hidden_dim = 48

    model = SimpleG(input_dim=input_size, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    bce = torch.nn.BCEWithLogitsLoss()
    bce_without_averaging = torch.nn.BCEWithLogitsLoss(reduction="sum")

    mfcc_converter = WaveToMFCCConverter(
        n_mfcc=input_size,
        sample_rate=dataset.sample_rate,
        win_length=dataset.label_window,
        hop_length=dataset.label_hop)

    train_dataloader.collate_fn = NoiseCollate(dataset.sample_rate, None, params, mfcc_converter)
    val_dataloader.collate_fn = ValidationCollate(dataset.sample_rate, None, val_params, val_snrs, mfcc_converter)

    if train_name == "":
        train_name = type(model).__name__

    model_trains_tree_dir = os.path.join(models_root_dir, train_name)

    if continue_last_model:
        model_path = find_last_model_in_tree(model_trains_tree_dir)
        if model_path is None:
            raise FileNotFoundError(f"Could not find model in folder {model_trains_tree_dir}")

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_epoch = checkpoint['epoch']

        input_size = checkpoint['model_input_size']
        hidden_dim = checkpoint['model_hidden_dim']

        mfcc_converter = WaveToMFCCConverter(
            n_mfcc=checkpoint['mfcc_n_mfcc'],
            sample_rate=checkpoint['mfcc_sample_rate'],
            win_length=checkpoint['mfcc_win_length'],
            hop_length=checkpoint['mfcc_hop_length'])

        print(f"Loaded {model_path} with optimizer {checkpoint['optimizer']}")
        print(f"Continuing training from epoch {global_epoch}")

    else:
        model_path = create_new_model_trains_dir(model_trains_tree_dir)
        global_epoch = 0
        print(f"Created {model_path}")

    loss_history_table = pd.DataFrame(columns=['global_epoch', 'train_loss'])
    accuracy_history_table = pd.DataFrame(columns=['global_epoch', 'train_accuracy'])
    loss_history_table.set_index('global_epoch', inplace=True)
    accuracy_history_table.set_index('global_epoch', inplace=True)

    for snr in val_snrs:
        if snr is None:
            loss_history_table['clear_audio_loss'] = []
            accuracy_history_table['clear_audio_accuracy'] = []
        else:
            loss_history_table[f'noised_audio_snr{snr}_loss'] = []
            accuracy_history_table[f'noised_audio_snr{snr}_accuracy'] = []

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
        time.sleep(0.2)
        model.train()
        for batch_inputs, batch_targets in tqdm(train_dataloader,
                                                desc=f"epoch {global_epoch + epoch}({epoch}\\{do_epoches})",
                                                disable=0):
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            output = model(batch_inputs)

            loss = bce(output, batch_targets)

            temp_count = batch_targets.numel()
            running_loss += loss.item() * temp_count
            running_whole_count += temp_count
            running_correct_count += torch.sum((output > threshold) == (batch_targets > threshold))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss /= running_whole_count
        accuracy = (running_correct_count / running_whole_count).item()

        row_loss_values = {
            'global_epoch': global_epoch + epoch,
            'train_loss': running_loss
        }
        row_acc_values = {
            'global_epoch': global_epoch + epoch,
            'train_accuracy': accuracy
        }

        time.sleep(0.5)
        print(f"{'=' * 40}")
        print("Training scores")
        print(f"Loss: {running_loss:.4f}\nAccuracy: {accuracy:.4f}")
        print(f"{'=' * 40}")

        model.eval()
        print("Getting validation scores")

        time.sleep(0.5)
        val_loss, val_acc = get_validation_score(model, bce_without_averaging, threshold, val_snrs,
                                                 val_dataloader, device)
        time.sleep(0.2)

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
                row_loss_values['clear_audio_loss'] = val_loss[snr].item()
                row_acc_values['clear_audio_accuracy'] = val_acc[snr].item()
            else:
                row_loss_values[f'noised_audio_snr{snr}_loss'] = val_loss[snr].item()
                row_acc_values[f'noised_audio_snr{snr}_accuracy'] = val_acc[snr].item()
        print(f"{'=' * 40}\n")

        loss_history_table.loc[len(loss_history_table)] = row_loss_values
        accuracy_history_table.loc[len(accuracy_history_table)] = row_acc_values

    torch.save({
        'epoch': global_epoch + do_epoches,
        'model_state_dict': model.state_dict(),
        'optimizer': type(optimizer).__name__,
        'optimizer_state_dict': optimizer.state_dict(),

        'model_input_size': input_size,
        'model_hidden_dim': hidden_dim,

        'mfcc_n_mfcc': mfcc_converter.n_mfcc,
        'mfcc_sample_rate': mfcc_converter.sample_rate,
        'mfcc_win_length': mfcc_converter.win_length,
        'mfcc_hop_length': mfcc_converter.hop_length,

    }, model_path)

    print(accuracy_history_table.T)
    print()
    print(f"{'=' * 40}\n")
    print(loss_history_table.T)
