import os
import random
import time

import pandas as pd
from tqdm import tqdm

import torch

from audio_utils import AudioWorker, OpenSLRDataset
from models import MODELS, NAMES
from utils import NoiseCollate, ValidationCollate, WaveToMFCCConverter
from utils import find_last_model_in_tree, create_new_model_trains_dir, get_train_val_dataloaders, print_as_table
from argument_parsers import train_parser

args = train_parser.parse_args()

noise_data_path = args.noise
clean_audios_path = args.clean
clean_labels_path = args.labels

if args.model_name is not None:
    model_name = args.model_name
    if model_name not in NAMES:
        raise ValueError(f"Model name must be one of: {NAMES}")
else:
    model_name = NAMES[args.model_id]

train_res_dir = args.train_res
load_last = args.use_last

batch_size = args.batch
num_workers = args.workers
val_batch_size = args.val_batch
val_num_workers = args.val_workers

train_ratio = 1 - args.val_ratio
do_epoches = args.epoch
epoch_noise_count = args.noise_pool
val_every = args.val_every
verbose = args.verbose

augmentation_params = {
    "noise_count": args.noise_count,
    "noise_duration_range": args.noise_duration,
    "snr_db": args.snr
}

val_params = augmentation_params.copy()
del val_params["snr_db"]
val_snrs = [None, 10, 5, 0]
threshold = args.threshold

if __name__ == '__main__':

    if args.snr in val_snrs:
        val_snrs.remove(args.snr)
    val_snrs.insert(0, args.snr)

    dataset = OpenSLRDataset(clean_audios_path, clean_labels_path, [])
    noise_files_paths = [os.path.join(noise_data_path, p) for p in os.listdir(noise_data_path) if p.endswith(".wav")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MODELS[model_name].to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    bce = torch.nn.BCEWithLogitsLoss()
    bce_without_averaging = torch.nn.BCEWithLogitsLoss(reduction="sum")

    model_trains_tree_dir = os.path.join(train_res_dir, model_name)

    model_new_dir, model_path = None, None
    if load_last or args.model_path is not None:
        if args.model_path is not None:
            model_path = args.model_path
            model_new_dir = os.path.dirname(model_path)
        else:
            model_new_dir, model_path = find_last_model_in_tree(model_trains_tree_dir)

    if model_path is not None:
        checkpoint = torch.load(model_path)

        seed = checkpoint["seed"]
        train_dataloader, val_dataloader, _ = get_train_val_dataloaders(dataset, train_ratio, batch_size,
                                                                        val_batch_size,
                                                                        num_workers, val_num_workers, seed)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        curr_run_start_global_epoch = checkpoint['epoch'] + 1

        mfcc_converter = WaveToMFCCConverter(
            n_mfcc=checkpoint['mfcc_n_mfcc'],
            sample_rate=checkpoint['mfcc_sample_rate'],
            win_length=checkpoint['mfcc_win_length'],
            hop_length=checkpoint['mfcc_hop_length'])

        loss_history_table = pd.read_csv(os.path.join(model_new_dir, 'loss_history.csv'), index_col="global_epoch")
        accuracy_history_table = pd.read_csv(os.path.join(model_new_dir, 'accuracy_history.csv'),
                                             index_col="global_epoch")

        print(f"Loaded {model_path} with optimizer {checkpoint['optimizer']}")
        print(f"Continuing training from epoch {curr_run_start_global_epoch + 1} on {device} device")

    else:
        print(f"Could not find model in folder {model_trains_tree_dir}")
        print(f"New model of {type(model)} type will be created instead")

        curr_run_start_global_epoch = 1

        train_dataloader, val_dataloader, seed = get_train_val_dataloaders(dataset, train_ratio, batch_size,
                                                                           val_batch_size,
                                                                           num_workers, val_num_workers)

        mfcc_converter = WaveToMFCCConverter(
            n_mfcc=model.input_dim,
            sample_rate=dataset.sample_rate,
            win_length=dataset.label_window,
            hop_length=dataset.label_hop)

        loss_history_table = pd.DataFrame(columns=['global_epoch', 'train_loss'])
        accuracy_history_table = pd.DataFrame(columns=['global_epoch', 'train_accuracy'])
        loss_history_table.set_index('global_epoch', inplace=True)
        accuracy_history_table.set_index('global_epoch', inplace=True)

        for snr in val_snrs:
            if snr is None:
                loss_history_table['clear_audio_loss'] = []
                accuracy_history_table['clear_audio_acc'] = []
            else:
                loss_history_table[f'noised_audio_snr{snr}_loss'] = []
                accuracy_history_table[f'noised_audio_snr{snr}_acc'] = []

    train_dataloader.collate_fn = NoiseCollate(dataset.sample_rate, None, augmentation_params, mfcc_converter)
    val_dataloader.collate_fn = ValidationCollate(dataset.sample_rate, None, val_params, val_snrs, mfcc_converter)

    for epoch in range(1, do_epoches + 1):

        global_epoch = curr_run_start_global_epoch + epoch - 1
        print(f"\n{'=' * 100}\n")

        noises = [AudioWorker(p, p.replace("\\", "__")) for p in random.sample(noise_files_paths, epoch_noise_count)]
        for noise in noises:
            noise.load()
            noise.resample(dataset.sample_rate)

        train_dataloader.collate_fn.noises = noises
        val_dataloader.collate_fn.noises = noises

        running_loss = torch.scalar_tensor(0, device=device)
        running_correct_count = torch.scalar_tensor(0, device=device)
        running_whole_count = torch.scalar_tensor(0, device=device)

        model.train()
        for batch_inputs, batch_targets in tqdm(train_dataloader,
                                                desc=f"Training epoch: {global_epoch} ({epoch}\\{do_epoches})",
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

        running_loss = (running_loss / running_whole_count).item()
        accuracy = (running_correct_count / running_whole_count).item()

        row_loss_values = {
            'global_epoch': global_epoch,
            'train_loss': running_loss
        }
        row_acc_values = {
            'global_epoch': global_epoch,
            'train_accuracy': accuracy
        }

        if verbose > 0:
            time.sleep(0.25)
            print(f"Training | loss: {running_loss:.4f} | accuracy: {accuracy:.4f}")
            time.sleep(0.25)

        val_loss, val_acc = None, None
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
                row_loss_values['clear_audio_loss'] = val_loss[snr].item() if val_loss is not None else None
                row_acc_values['clear_audio_acc'] = val_acc[snr].item() if val_acc is not None else None
            else:
                row_loss_values[f'noised_audio_snr{snr}_loss'] = val_loss[snr].item() if val_loss is not None else None
                row_acc_values[f'noised_audio_snr{snr}_acc'] = val_acc[snr].item() if val_acc is not None else None

        loss_history_table.loc[global_epoch] = row_loss_values
        accuracy_history_table.loc[global_epoch] = row_acc_values

        if verbose > 1:
            print(f"\nLoss history")
            print_as_table(loss_history_table)
            print(f"\nAccuracy history")
            print_as_table(accuracy_history_table)

    model_new_dir, model_path = create_new_model_trains_dir(model_trains_tree_dir)
    print(f"\nCreated {model_new_dir}")

    torch.save({
        'seed': seed,
        'epoch': curr_run_start_global_epoch + do_epoches - 1,
        'model_state_dict': model.state_dict(),
        'optimizer': type(optimizer).__name__,
        'optimizer_state_dict': optimizer.state_dict(),

        'mfcc_n_mfcc': mfcc_converter.n_mfcc,
        'mfcc_sample_rate': mfcc_converter.sample_rate,
        'mfcc_win_length': mfcc_converter.win_length,
        'mfcc_hop_length': mfcc_converter.hop_length,

    }, model_path)

    loss_history_table.to_csv(os.path.join(model_new_dir, 'loss_history.csv'))
    accuracy_history_table.to_csv(os.path.join(model_new_dir, 'accuracy_history.csv'))

    if not args.no_plot:
        loss_plot = loss_history_table.plot()
        accuracy_plot = accuracy_history_table.plot()

        loss_plot.figure.savefig(os.path.join(model_new_dir, 'loss.png'))
        accuracy_plot.figure.savefig(os.path.join(model_new_dir, 'accuracy.png'))

    print(f"Saved as {model_path}")
