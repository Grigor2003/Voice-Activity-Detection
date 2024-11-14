import os
import random
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import auc

import torch
from torch.utils.data import DataLoader

from other.audio_utils import AudioWorker, OpenSLRDataset, EnotDataset
from models_handler import MODELS
from other.utils import NoiseCollate, WaveToMFCCConverter
from other.utils import find_model_in_dir_or_path
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

    checkpoint = torch.load(weights_path, weights_only=True)

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

    dataloader.collate_fn = NoiseCollate(dataset.sample_rate, augmentation_params, snr, mfcc_converter)

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

    for (batch_inputs, mask, batch_targets), _ in tqdm(dataloader):
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        mask = mask.to(device)

        out = model(batch_inputs, ~mask)
        output = mask * out.squeeze(-1)

        for i, thold in enumerate(thresholds):
            positive = batch_targets.to(torch.bool)
            pred = output > thold
            tp_to_thold[i] += torch.sum(torch.logical_and(pred, positive) * mask)
            tn_to_thold[i] += torch.sum(torch.logical_and(~pred, ~positive) * mask)
            fn_to_thold[i] += torch.sum(torch.logical_and(~pred, positive) * mask)
            fp_to_thold[i] += torch.sum(torch.logical_and(pred, ~positive) * mask)

    roc_x = (fp_to_thold / (fp_to_thold + tn_to_thold)).cpu()
    roc_y = (tp_to_thold / (tp_to_thold + fn_to_thold)).cpu()
    roc_auc = auc(roc_x, roc_y)

    recall = roc_y
    precision = (tp_to_thold / (tp_to_thold + fp_to_thold)).cpu().nan_to_num(0)
    f1_score = (2 * precision * recall / (precision + recall)).nan_to_num(0)

    best_th_by_f1 = thresholds[np.argmax(f1_score)]

    plt.figure()
    plt.plot(roc_x, roc_y, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.005])
    plt.xlabel('False Positive Rate'),
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    plt.figure()
    plt.plot(thresholds, f1_score, label=f'F1')
    plt.plot(thresholds, precision, label=f'Precision')
    plt.plot(thresholds, recall, label=f'Recall')
    plt.vlines(best_th_by_f1, 0, 1, color='gray', linestyle='--', label=f'Best threshold on {best_th_by_f1} by f1')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.005])
    plt.xlabel('Thresholds')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
