if __name__ == '__main__':
    import os
    import random
    import time
    import shutil

    import pandas as pd
    from tqdm import tqdm

    import torch

    from other.audio_utils import AudioWorker, OpenSLRDataset
    from models_handler import MODELS, count_parameters, estimate_vram_usage
    from other.utils import NoiseCollate, ValCollate, WaveToMFCCConverter
    from other.utils import find_last_model_in_tree, create_new_model_trains_dir, get_train_val_dataloaders, \
        print_as_table, \
        save_history_plot, find_model_in_dir_or_path
    from other.train_args_parser import *

    print(f"\n{'=' * 100}\n")
    dataset = OpenSLRDataset(clean_audios_path, clean_labels_path)
    noise_files_paths = [os.path.join(noise_data_path, p) for p in os.listdir(noise_data_path) if p.endswith(".wav")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MODELS[model_name]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=10 ** lr)
    bce = torch.nn.BCELoss(reduction="sum")

    model_trains_tree_dir = os.path.join(train_res_dir, model_name)
    model_dir, model_path = None, None
    last_weights_path = None

    if load_from is not None:
        last_weights_path = find_model_in_dir_or_path(load_from)
    elif load_last:
        model_dir, model_path = find_last_model_in_tree(model_trains_tree_dir)
        if model_path is None:
            print(f"Couldn't find any model in {model_trains_tree_dir} so new model will be created")
        else:
            last_weights_path = model_path

    train_dataloader, val_dataloader, seed, mfcc_converter = [None] * 4
    if last_weights_path is not None:
        checkpoint = torch.load(last_weights_path, weights_only=True)

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

        print(f"Successfully loaded {last_weights_path} with optimizer {checkpoint['optimizer']}")
        print(f"Continuing training from epoch {curr_run_start_global_epoch} on {device} device")
    else:
        curr_run_start_global_epoch = 1
        print(f"New model of {str(type(model))} type has been created, will be trained on {device} device")

    print(
        f"This model has estimated {count_parameters(model)} parameters, and costs {estimate_vram_usage(model)} GB while float32")

    try:
        loss_history_table = pd.read_csv(os.path.join(model_dir, 'loss_history.csv'), index_col="global_epoch")
        accuracy_history_table = pd.read_csv(os.path.join(model_dir, 'accuracy_history.csv'),
                                             index_col="global_epoch")
    except:

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

        for snr in val_snrs_list:
            if snr is None:
                loss_history_table['clear_audio_loss'] = []
                accuracy_history_table['clear_audio_acc'] = []
            else:
                loss_history_table[f'noised_audio_snr{snr}_loss'] = []
                accuracy_history_table[f'noised_audio_snr{snr}_acc'] = []

    train_dataloader.collate_fn = NoiseCollate(dataset.sample_rate, aug_params, snr_dict, mfcc_converter)
    val_dataloader.collate_fn = ValCollate(dataset.sample_rate, aug_params, val_snrs_list, mfcc_converter)

    print(f"Checkpoints(for this run): {save_frames}")

    print(f"Training [SNR values: {', '.join(map(str, snr_dict))}, " +
          f"final batch size: {batch_size}]")
    print(f"Validation [SNR values: {', '.join(map(str, val_snrs_list))}, " +
          f"Batch size: {len(val_snrs_list) * val_batch_size}]")

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
        for batch_inputs, mask, batch_targets in tqdm(train_dataloader,
                                                      desc=f"Training epoch: {global_epoch} ({epoch}\\{do_epoches})" + ' | ',
                                                      disable=0):
            batch_inputs = batch_inputs.to(device)
            mask = mask.to(device)
            batch_targets = batch_targets.to(device)

            out = model(batch_inputs)
            output = mask * out.squeeze(-1)

            real_samples_count = mask.sum()
            loss = bce(output, batch_targets) / real_samples_count

            running_loss += loss.item() * real_samples_count
            running_whole_count += real_samples_count
            running_correct_count += torch.sum(((output > threshold) == (batch_targets > threshold)) * mask)

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

        if print_level > 0:
            time.sleep(0.25)
            print(f"Training | loss: {running_loss:.4f} | accuracy: {accuracy:.4f}")
            time.sleep(0.25)

        val_loss, val_acc = None, None
        if val_every != 0 and epoch % val_every == 0:
            model.eval()

            with torch.no_grad():
                val_loss = {snr_db: torch.scalar_tensor(0.0, device=device) for snr_db in val_snrs_list}
                val_acc = {snr_db: torch.scalar_tensor(0.0, device=device) for snr_db in val_snrs_list}
                correct_count = {snr_db: 0 for snr_db in val_snrs_list}
                whole_count = {snr_db: 0 for snr_db in val_snrs_list}

                print()
                time.sleep(0.25)
                for all_tensors in tqdm(val_dataloader, desc=f"Calculating validation scores: "):
                    for snr_db in val_snrs_list:
                        batch_inputs = all_tensors[snr_db][0].to(device)
                        mask = all_tensors[snr_db][1].to(device)
                        batch_targets = all_tensors[snr_db][2].to(device)
                        real_samples_count = mask.sum()

                        output = mask * model(batch_inputs).squeeze(-1)
                        val_loss[snr_db] += bce(output, batch_targets)
                        correct_count[snr_db] += torch.sum(((output > threshold) == (batch_targets > threshold)) * mask)
                        whole_count[snr_db] += real_samples_count

                for snr_db in val_snrs_list:
                    val_loss[snr_db] /= whole_count[snr_db]
                    val_acc[snr_db] = correct_count[snr_db] / whole_count[snr_db]

        for snr in val_snrs_list:
            if snr is None:
                row_loss_values['clear_audio_loss'] = val_loss[snr].item() if val_loss is not None else np.nan
                row_acc_values['clear_audio_acc'] = val_acc[snr].item() if val_acc is not None else np.nan
            else:
                row_loss_values[f'noised_audio_snr{snr}_loss'] = val_loss[
                    snr].item() if val_loss is not None else np.nan
                row_acc_values[f'noised_audio_snr{snr}_acc'] = val_acc[snr].item() if val_acc is not None else np.nan

        loss_history_table.loc[global_epoch] = row_loss_values
        accuracy_history_table.loc[global_epoch] = row_acc_values

        if val_every != 0 and print_level > 1 and epoch % val_every == 0:
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
            print()
            print(f"Model saved (global epoch: {global_epoch}, checkpoint: {epoch})")
