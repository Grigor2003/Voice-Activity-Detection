import os
import random
import time

import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, random_split

from audio_utils import AudioWorker, OpenSLRDataset
from models import MODELS, NAMES
from utils import NoiseCollate, ValidationCollate, WaveToMFCCConverter
from utils import find_last_model_in_tree, create_new_model_trains_dir, get_validation_score
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

    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=val_num_workers)

    model = MODELS[model_name].to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    bce_without_averaging = torch.nn.BCEWithLogitsLoss(reduction="sum")

    model_trains_tree_dir = os.path.join(train_res_dir, model_name)

    if load_last or args.model_path is not None:
        if args.model_path is not None:
            model_path = args.model_path
            model_new_dir = os.path.dirname(model_path)
        else:
            model_new_dir, model_path = find_last_model_in_tree(model_trains_tree_dir)

        if model_path is None:
            raise FileNotFoundError(f"Could not find model in folder {model_trains_tree_dir}")

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_epoch = checkpoint['epoch']

        mfcc_converter = WaveToMFCCConverter(
            n_mfcc=checkpoint['mfcc_n_mfcc'],
            sample_rate=checkpoint['mfcc_sample_rate'],
            win_length=checkpoint['mfcc_win_length'],
            hop_length=checkpoint['mfcc_hop_length'])

        loss_history_table = pd.read_csv(os.path.join(model_new_dir, 'loss_history.csv'), index_col="global_epoch")
        accuracy_history_table = pd.read_csv(os.path.join(model_new_dir, 'accuracy_history.csv'),
                                             index_col="global_epoch")

        print(f"Loaded {model_path} with optimizer {checkpoint['optimizer']}")
        print(f"Continuing training from epoch {global_epoch}")

    else:
        global_epoch = 0

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
                accuracy_history_table['clear_audio_accuracy'] = []
            else:
                loss_history_table[f'noised_audio_snr{snr}_loss'] = []
                accuracy_history_table[f'noised_audio_snr{snr}_accuracy'] = []

    train_dataloader.collate_fn = NoiseCollate(dataset.sample_rate, None, augmentation_params, mfcc_converter)
    val_dataloader.collate_fn = ValidationCollate(dataset.sample_rate, None, val_params, val_snrs, mfcc_converter)

    for epoch in range(1, do_epoches + 1):
        noises = [AudioWorker(p, p.replace("\\", "__")) for p in random.sample(noise_files_paths, epoch_noise_count)]
        for noise in noises:
            noise.load()
            noise.resample(dataset.sample_rate)

        train_dataloader.collate_fn.noises = noises
        val_dataloader.collate_fn.noises = noises

        running_loss = torch.scalar_tensor(0, device=device)
        running_correct_count = torch.scalar_tensor(0, device=device)
        running_whole_count = torch.scalar_tensor(0, device=device)

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

        running_loss = (running_loss / running_whole_count).item()
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

        if verbose > 0:
            print(f"{'=' * 40}")
            print("Training scores")
            print(f"Loss: {running_loss:.4f}\nAccuracy: {accuracy:.4f}")
            print(f"{'=' * 40}")

        if epoch % val_every == 0:
            model.eval()
            print("Getting validation scores")

            time.sleep(0.5)
            val_loss, val_acc = get_validation_score(model, bce_without_averaging, threshold, val_snrs,
                                                     val_dataloader, device)
            time.sleep(0.2)

            if verbose > 1:
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

    model_new_dir, model_path = create_new_model_trains_dir(model_trains_tree_dir)
    print(f"Created {model_new_dir}")

    torch.save({
        'epoch': global_epoch + do_epoches,
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
    print()
    print(accuracy_history_table.T)
    print()
    print(f"{'=' * 40}\n")
    print(loss_history_table.T)
