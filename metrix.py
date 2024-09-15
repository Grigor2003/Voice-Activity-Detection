import os
import random
import time
import shutil

import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import auc

import torch
from torch.utils.data import DataLoader

from other.audio_utils import AudioWorker, OpenSLRDataset, EnotDataset
from models_handler import MODELS
from other.utils import NoiseCollate, ValidationCollate, WaveToMFCCConverter
from other.utils import create_new_model_trains_dir, get_train_val_dataloaders, print_as_table, \
    save_history_plot, find_model_in_dir_or_path
from other.metrix_args_parser import *

if __name__ == '__main__':

    print(f"\n{'=' * 100}\n")
    dataset = OpenSLRDataset(clean_audios_path, clean_labels_path)
    noise_files_paths = [os.path.join(noise_data_path, p) for p in os.listdir(noise_data_path) if p.endswith(".wav")]
    if enot_data_path is not None:
        enot_dataset = EnotDataset(enot_data_path, dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MODELS[model_name]().to(device)
    model.eval()

    weights_path = find_model_in_dir_or_path(load_from)

    checkpoint = torch.load(weights_path)

    seed = checkpoint["seed"]

    model.load_state_dict(checkpoint['model_state_dict'])
    global_epoch = checkpoint['epoch'] + 1

    mfcc_converter = WaveToMFCCConverter(
        n_mfcc=checkpoint['mfcc_n_mfcc'],
        sample_rate=checkpoint['mfcc_sample_rate'],
        win_length=checkpoint['mfcc_win_length'],
        hop_length=checkpoint['mfcc_hop_length'])

    # TODO: stugel ardyoq checkpointi samplerate win hop hamynknum en labelneri het te che

    print(f"Successfully loaded {weights_path}")
    print(f"Metrix will be applied to {global_epoch} epoches trained model on {device} device")

    dataloader.collate_fn = NoiseCollate(dataset.sample_rate, None, augmentation_params, mfcc_converter)

    print(f"\n{'=' * 100}\n")

    noises = [AudioWorker(p, p.replace("\\", "__")) for p in random.sample(noise_files_paths, epoch_noise_count)]
    for noise in noises:
        noise.load()
        noise.resample(dataset.sample_rate)

    dataloader.collate_fn.noises = noises

    whole_count = torch.scalar_tensor(0, device=device)
    tp_to_thold = torch.tensor([0] * len(thresholds), dtype=torch.float32, device=device)
    fn_to_thold = torch.tensor([0] * len(thresholds), dtype=torch.float32, device=device)
    fp_to_thold = torch.tensor([0] * len(thresholds), dtype=torch.float32, device=device)
    tn_to_thold = torch.tensor([0] * len(thresholds), dtype=torch.float32, device=device)

    for batch_inputs, batch_targets in tqdm(dataloader):
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        output = model(batch_inputs)

        temp_count = batch_targets.numel()
        whole_count += temp_count
        for i, thold in enumerate(thresholds):
            positive = batch_targets.to(torch.bool)
            pred = output > thold
            tp_to_thold[i] += torch.sum(torch.logical_and(pred, positive))
            tn_to_thold[i] += torch.sum(torch.logical_and(~pred, ~positive))
            fn_to_thold[i] += torch.sum(torch.logical_and(~pred, positive))
            fp_to_thold[i] += torch.sum(torch.logical_and(pred, ~positive))

    roc_x = (fp_to_thold / (fp_to_thold + tn_to_thold)).cpu()
    roc_y = (tp_to_thold / (tp_to_thold + fn_to_thold)).cpu()
    roc_auc = auc(roc_x, roc_y)

    plt.figure()
    plt.plot(roc_x, roc_y, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.005])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    row_loss_values = {
        'global_epoch': global_epoch,
        'train_loss': running_loss
    }
    row_acc_values = {
        'global_epoch': global_epoch,
        'train_accuracy': accuracy
    }

    if print_level > 0:
        time.sleep(0.25)
        print(f"Training | loss: {running_loss:.4f} | accuracy: {accuracy:.4f}")
        time.sleep(0.25)

    val_loss, val_acc = None, None
    if val_every != 0:
        if epoch % val_every == 0:
            model.eval()

            val_loss = {snr_db: torch.scalar_tensor(0.0, device=device) for snr_db in val_snrs}
            val_acc = {snr_db: torch.scalar_tensor(0.0, device=device) for snr_db in val_snrs}
            correct_count = {snr_db: 0 for snr_db in val_snrs}
            whole_count = {snr_db: 0 for snr_db in val_snrs}

            for all_tensors in tqdm(val_dataloader, desc=f"Calculating validation scores: "):
                for snr_db in val_snrs:
                    batch_inputs = all_tensors[snr_db][0].to(device)
                    batch_targets = all_tensors[snr_db][1].to(device)
                    output = model(batch_inputs)
                    val_loss[snr_db] += bce_without_averaging(output, batch_targets)
                    correct_count[snr_db] += torch.sum((output > threshold) == (batch_targets > threshold))
                    whole_count[snr_db] += batch_targets.numel()

            for snr_db in val_snrs:
                val_loss[snr_db] /= whole_count[snr_db]
                val_acc[snr_db] = correct_count[snr_db] / whole_count[snr_db]

    for snr in val_snrs:
        if snr is None:
            row_loss_values['clear_audio_loss'] = val_loss[snr].item() if val_loss is not None else np.nan
            row_acc_values['clear_audio_acc'] = val_acc[snr].item() if val_acc is not None else np.nan
        else:
            row_loss_values[f'noised_audio_snr{snr}_loss'] = val_loss[
                snr].item() if val_loss is not None else np.nan
            row_acc_values[f'noised_audio_snr{snr}_acc'] = val_acc[snr].item() if val_acc is not None else np.nan

    loss_history_table.loc[global_epoch] = row_loss_values
    accuracy_history_table.loc[global_epoch] = row_acc_values

    if print_level > 1:
        print(f"\nLoss history")
        print_as_table(loss_history_table)
        print(f"\nAccuracy history")
        print_as_table(accuracy_history_table)

    if epoch in save_frames:
        if model_dir is None:
            model_dir, model_path = create_new_model_trains_dir(model_trains_tree_dir)
            print(f"\nCreated {model_dir}")

        if last_weights_path is not None:
            old = os.path.join(model_dir, "old")
            os.makedirs(old, exist_ok=True)
            shutil.copy(last_weights_path, old)

        loss_history_table.to_csv(os.path.join(model_dir, 'loss_history.csv'))
        accuracy_history_table.to_csv(os.path.join(model_dir, 'accuracy_history.csv'))

        if plot:
            save_history_plot(loss_history_table, 'global_epoch', 'Loss history', 'Epoch', 'Loss',
                              os.path.join(model_dir, 'loss.png'))

            save_history_plot(accuracy_history_table, 'global_epoch', 'Accuracy history', 'Epoch', 'Accuracy',
                              os.path.join(model_dir, 'accuracy.png'))

        torch.save({
            'seed': seed,
            'epoch': global_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer': type(optimizer).__name__,
            'optimizer_state_dict': optimizer.state_dict(),

            'mfcc_n_mfcc': mfcc_converter.n_mfcc,
            'mfcc_sample_rate': mfcc_converter.sample_rate,
            'mfcc_win_length': mfcc_converter.win_length,
            'mfcc_hop_length': mfcc_converter.hop_length,

        }, model_path)
        last_weights_path = model_path
        print(f"Model saved (global epoch: {global_epoch}, checkpoint: {epoch})")