import torch
import torchaudio
import os
import time
import shutil
import pandas as pd
from tqdm import tqdm

from other.data.audio_utils import AudioWorker
from other.models.models_handler import MODELS, count_parameters, estimate_vram_usage
from other.data.collates import NoiseCollate, ValCollate
from other.data.datasets import CommonAccent
from other.data.processing import get_train_val_dataloaders, WaveToMFCCConverter
from other.utils import EXAMPLE_FOLDER, loss_function, async_message_box
from other.utils import find_last_model_in_tree, create_new_model_trains_dir, find_model_in_dir_or_path
from other.utils import print_as_table, save_history_plot

if __name__ == '__main__':
    from other.parsing.train_args_parser import *

    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # os.environ['TORCH_USE_CUDA_DSA'] = "1"
    info_txt = ""
    info_txt += '\n' + f"Create new model: {create_new_model}"

    model_dir, model_path = None, None
    last_weights_path = None
    pretrained_vad_checkpoint = None

    if weights_load_from is not None:
        last_weights_path = find_model_in_dir_or_path(weights_load_from)
    else:
        if create_new_model is None:
            model_dir, last_weights_path = find_last_model_in_tree(model_name)
        elif not create_new_model:
            model_dir, model_path = find_last_model_in_tree(model_name)
            last_weights_path = model_path
            if model_path is None:
                # noinspection PyRedundantParentheses
                info_txt += '\n' + (
                    f"WARNING : Couldn't find weights in {model_name} so brand new model will be created")
        elif create_new_model and pretrained_vad is not None:
            pretrained_vad_checkpoint = torch.load(pretrained_vad, weights_only=True)

    checkpoint = None
    if last_weights_path is not None:
        checkpoint = torch.load(last_weights_path, weights_only=True)

    curr_run_start_global_epoch = 1
    if checkpoint is not None:
        try:
            curr_run_start_global_epoch = checkpoint['epoch'] + 1
        except:
            curr_run_start_global_epoch = torch.nan
            # noinspection PyRedundantParentheses
            info_txt += '\n' + (f"WARNING : Last train epochs count couldn't be found in the checkpoint")
    # noinspection PyRedundantParentheses
    info_txt += '\n' + (f"Global epoch : {curr_run_start_global_epoch}")

    seed = seed  # Needs for PyCharm's satisfaction
    if checkpoint is not None:
        try:
            seed = checkpoint["seed"]
        except:
            # noinspection PyRedundantParentheses
            info_txt += '\n' + (f"WARNING : Last train seed couldn't be found in the checkpoint")
    torch.manual_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    # noinspection PyRedundantParentheses
    info_txt += '\n' + (f"Global seed : {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # noinspection PyRedundantParentheses
    info_txt += '\n' + (f"Device : {device}")

    model = MODELS[model_name]().to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    if pretrained_vad_checkpoint:
        model.load_state_dict(pretrained_vad_checkpoint['model_state_dict'], strict=False)
        # noinspection PyRedundantParentheses
        info_txt += '\n' + (f"Pretrained weights : {pretrained_vad}")
    # noinspection PyRedundantParentheses
    info_txt += '\n' + (f"Model : {model_name}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if checkpoint is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            optimizer.lr = lr
        except:
            # noinspection PyRedundantParentheses
            info_txt += '\n' + (f"WARNING : Couldn't load optimizer states from the checkpoint")
    # noinspection PyRedundantParentheses
    info_txt += '\n' + (f"Optimizer : {type(optimizer)}")

    dataset = CommonAccent(accent_dir)
    info_txt += '\n' + ("Train dataset" +
                        f"\n\t- files count : {len(dataset)}" +
                        f"\n\t- labels dir: '{accent_dir}'")

    noise_files_paths = [os.path.join(noise_data_path, p) for p in os.listdir(noise_data_path) if p.endswith(".wav")]
    info_txt += '\n' + ("Noise files" +
                        f"\n\t- files count : {len(noise_files_paths)}" +
                        f"\n\t- path: '{noise_data_path}'")

    train_dataloader, val_dataloader = get_train_val_dataloaders(dataset, train_ratio, batch_size, val_batch_size,
                                                                 num_workers, val_num_workers, generator)

    if checkpoint is not None:
        mfcc_converter = WaveToMFCCConverter(
            n_mfcc=checkpoint['mfcc_n_mfcc'],
            sample_rate=checkpoint['mfcc_sample_rate'],
            win_length=checkpoint['mfcc_win_length'],
            hop_length=checkpoint['mfcc_hop_length'])
    else:
        mfcc_converter = WaveToMFCCConverter(
            n_mfcc=model.input_dim,
            sample_rate=dataset.sample_rate,
            win_length=default_win_length,
            hop_length=default_win_length // 2)

    info_txt += '\n' + ("Estimated" +
                        f"\n\t- parameters : {count_parameters(model)}" +
                        f"\n\t- VRAM: {estimate_vram_usage(model):.4f} GB (if float32)")

    try:
        loss_history_table = pd.read_csv(os.path.join(model_dir, 'loss_history.csv'), index_col="global_epoch")
        accuracy_history_table = pd.read_csv(os.path.join(model_dir, 'accuracy_history.csv'),
                                             index_col="global_epoch")
    except:
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

    train_dataloader.collate_fn = NoiseCollate(dataset.sample_rate, aug_params, snr_dict, mfcc_converter, zero_count)
    val_dataloader.collate_fn = ValCollate(dataset.sample_rate, aug_params, val_snrs_list, mfcc_converter)

    # noinspection PyRedundantParentheses
    info_txt += '\n' + (f"Checkpoints : {saves_count} in {do_epoches}")
    info_txt += '\n' + ("Training" +
                        f"\n\t- SNR values : [{', '.join(map(str, snr_dict))}]".replace('None', '_') +
                        f"\n\t- final batch size:  {batch_size + zero_count if zero_count is not None else 0}")

    if val_every != 0:
        info_txt += '\n' + ("Validation" +
                            f"\n\t- SNR values : [{', '.join(map(str, val_snrs_list))}]".replace('None', '_') +
                            f"\n\t- batch size:  {len(val_snrs_list) * val_batch_size}")
    else:
        # noinspection PyRedundantParentheses
        info_txt += '\n' + ("Validation : No validation is expecting for this run")
    if print_mbox:
        if last_weights_path is not None:
            path = os.path.normpath(last_weights_path)
            loc = path.split(os.sep)[-3:-1]
            async_message_box(f"{model_name} training in {loc}", info_txt, 0)
        else:
            async_message_box(f"New {model_name} training", info_txt, 0)

    print(f"\n{'=' * 100}\n")
    print("Training")

    for epoch in range(1, do_epoches + 1):

        global_epoch = curr_run_start_global_epoch + epoch - 1
        print(f"\n{'=' * 100}\n")

        inds = torch.randperm(len(noise_files_paths))[:epoch_noise_count]

        noises = [AudioWorker(noise_files_paths[i]) for i in inds]
        for noise in noises:
            noise.load()
            noise.resample(dataset.sample_rate)

        train_dataloader.collate_fn.noises = noises
        val_dataloader.collate_fn.noises = noises

        running_loss = torch.scalar_tensor(0, device=device)
        running_correct_count = torch.scalar_tensor(0, device=device)
        running_whole_count = torch.scalar_tensor(0, device=device)

        model.train()
        batch_idx, batch_count = 0, len(train_dataloader)

        for batch_idx, (batch_inputs, mask, batch_targets) in enumerate(
                tqdm(train_dataloader, desc=f"Training epoch: {global_epoch} ({epoch}\\{do_epoches})" + ' | ',
                     disable=0)):

            # Move data to the GPU if available
            batch_inputs = batch_inputs.to(device)
            mask = mask.to(device)
            batch_targets = batch_targets.to(device)

            # Forward pass
            output = model(batch_inputs, ~mask)

            # Calculate the loss
            loss = torch.nn.CrossEntropyLoss()(output, torch.argmax(batch_targets, dim=1))
            # loss = loss_function(output, batch_targets, mask)
            loss = loss / accumulation_steps  # Scale loss by the number of accumulation steps

            # Accumulate running loss and correct count (for logging/metrics)
            batch_samples_count = mask.size(0)
            running_loss += loss.item() * batch_samples_count * accumulation_steps  # Rescale back for logging
            running_whole_count += batch_samples_count
            pred_correct = torch.argmax(output, dim=1) == torch.argmax(batch_targets, dim=1)
            running_correct_count += torch.sum(torch.sum(pred_correct, dim=-1) / mask.sum(dim=-1))

            # Backward pass (accumulate gradients)
            loss.backward()

            # Perform optimizer step and zero gradients every `accumulation_steps` batches
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # After the final batch, check if there are remaining gradients to update
        if (batch_idx + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

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
                    for snr_db, (batch_inputs, mask, batch_targets) in all_tensors.items():
                        batch_inputs = batch_inputs.to(device)
                        mask = mask.to(device)
                        batch_targets = batch_targets.to(device)
                        real_samples_count = mask.size(0)

                        output = mask * model(batch_inputs, ~mask).squeeze(-1)
                        val_loss[snr_db] += loss_function(output, batch_targets, mask, val=True).item()

                        pred_correct = ((output > threshold) == (batch_targets > threshold)) * mask
                        correct_count[snr_db] += torch.sum(torch.sum(pred_correct, dim=-1) / mask.sum(dim=-1))
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

        if print_val_results and val_every != 0 and epoch % val_every == 0:
            print(f"\nLoss history")
            print_as_table(loss_history_table)
            print(f"\nAccuracy history")
            print_as_table(accuracy_history_table)

        if epoch in save_frames:
            if model_path is None:
                model_dir, model_path = create_new_model_trains_dir(model_name)
                print(f"\nCreated {model_dir}")

            curr_info_save_path = os.path.join(model_dir, 'info.txt')
            with open(curr_info_save_path, 'w') as f:
                f.write(info_txt)

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
            print(f"\nModel saved (global epoch: {global_epoch}, checkpoint: {epoch})")

            last_weights_path = model_path
            model_has_been_saved()

    print()
    print(info_txt)
