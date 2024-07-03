import os
import random
from datetime import datetime
from tqdm import tqdm
import torch

from torch.utils.data import DataLoader
from audio_utils import AudioWorker, OpenSLRDataset
from gru_model import SimpleG
from utils import NoiseCollate, WaveToMFCCConverter, find_last_model_in_tree, create_new_model_dir

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

    if continue_last_model:
        model_path = find_last_model_in_tree(train_name)

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_epoch = checkpoint['epoch']
        model.train()
        print(f"Loaded {model_path}")
        print(f"Continuing training from epoch {global_epoch}")

    else:
        model_path = create_new_model_dir(train_name)
        global_epoch = 0
        print(f"Created {model_path}")

    for epoch in range(1, do_epoches + 1):
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
            'epoch': global_epoch + epoch,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.__ne__,
            'optimizer_state_dict': optimizer.state_dict()
        }, model_path)

    print(loss_history)
