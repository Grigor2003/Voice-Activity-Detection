import os
from datetime import datetime
import glob
import ctypes
import threading

import numpy as np
import torch
from matplotlib import pyplot as plt
from tabulate import tabulate

RES_FOLDER = "RESULTS"
BRAND_MODEL_PREFIX = "START"
RUN_PREFIX = "run"
DATE_FORMAT = "%Y-%m-%d"
MODEL_NAME = "weights"
MODEL_EXT = ".pt"
EXAMPLE_FOLDER = "examples"


def loss_function(pred, target, mask, reduction="auto", val=False):
    ce_loss = torch.nn.functional.binary_cross_entropy(pred, target, reduction='none')
    masked_loss = ce_loss * mask
    loss = torch.sum(masked_loss, dim=-1) / mask.sum(dim=-1).float()
    loss = loss ** 2

    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    elif reduction == "auto":
        loss = loss.sum() if val else loss.mean()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss


class Example:
    def __init__(self,
                 wave: torch.Tensor = None,
                 clear: torch.Tensor = None,
                 label: torch.Tensor = None,
                 pred: torch.Tensor = None,
                 name: str = None, info_dicts: list[dict] = None,
                 bi: int = None, i: int = None):
        self.wave = wave
        self.clear = clear
        self.label = label
        self.pred = pred
        self.name = name
        self.info_dicts = info_dicts
        self.bi = bi
        self.i = i

    def update(self,
               wave: torch.Tensor = None,
               clear: torch.Tensor = None,
               label: torch.Tensor = None,
               pred: torch.Tensor = None,
               name: str = None, info_dicts: list[dict] = None,
               bi: int = None, i: int = None):
        if wave is not None:
            self.wave = wave
        if clear is not None:
            self.clear = clear
        if label is not None:
            self.label = label
        if pred is not None:
            self.pred = pred
        if name is not None:
            self.name = name
        if info_dicts is not None:
            self.info_dicts = info_dicts
        if bi is not None:
            self.bi = bi
        if i is not None:
            self.i = i


def print_as_table(dataframe):
    if len(dataframe) > 4:
        print(tabulate(dataframe.iloc[[0, -3, -2, -1], :].T.fillna("---"), headers='keys'))
    else:
        print(tabulate(dataframe.T.fillna("---"), headers='keys'))


def get_files_by_extension(directory, ext='txt', rel=False):
    ext = ext if ext.startswith('.') else ('.' + ext)
    pattern = os.path.join(directory, '**', f'*{ext}')
    files = glob.glob(pattern, recursive=True)
    if rel:
        return [os.path.relpath(path, directory) for path in files]
    return files


def change_file_extension(file_path, new_extension):
    ext = new_extension.strip('.')
    return os.path.splitext(file_path)[0] + "." + ext


def find_model_in_dir_or_path(dp: str):
    if os.path.isdir(dp):
        for file in os.listdir(dp):
            if file.endswith(".pt"):
                return os.path.join(dp, file)
        raise FileNotFoundError(f"There is no model file in given directory: {dp}")
    elif os.path.isfile(dp):
        if dp.endswith(".pt"):
            return dp
        raise TypeError(f"Model file must be pytorch model: {dp}")


def find_last_model_in_tree(model_name):
    res_dir = None

    model_trains_tree_dir = os.path.join(RES_FOLDER, model_name)

    if os.path.exists(model_trains_tree_dir):
        max_num = 0
        max_name = None
        for name in os.listdir(model_trains_tree_dir):
            bnm, num, _ = name.split("_")
            if bnm == BRAND_MODEL_PREFIX:
                if int(num) >= max_num:
                    max_num = int(num)
                    max_name = name
        if max_name is not None:
            brand_dir = os.path.join(model_trains_tree_dir, max_name)
            max_num = 0
            for name in os.listdir(brand_dir):
                num = int(name.split("_")[1])
                folder_path = os.path.join(brand_dir, name)
                if num >= max_num and (MODEL_NAME + MODEL_EXT) in os.listdir(folder_path):
                    max_num = num
                    res_dir = folder_path

    if res_dir is None:
        return None, None
    else:
        return res_dir, os.path.join(res_dir, (MODEL_NAME + MODEL_EXT))


def create_new_model_trains_dir(model_name, brand_new=False):
    model_trains_tree_dir = os.path.join(RES_FOLDER, model_name)
    os.makedirs(model_trains_tree_dir, exist_ok=True)

    max_num = 0
    for name in os.listdir(model_trains_tree_dir):
        num = int(name.split("_")[1])
        max_num = max(num, max_num)

    take = max_num + 1 if brand_new else max_num
    brand_name = BRAND_MODEL_PREFIX + f"_{take}" + f"_({datetime.now().strftime(DATE_FORMAT)})"
    brand_dir = os.path.join(model_trains_tree_dir, brand_name)
    os.makedirs(brand_dir, exist_ok=True)

    max_num = 0
    for name in os.listdir(brand_dir):
        num = int(name.split("_")[1])
        max_num = max(num, max_num)

    run_dir = os.path.join(brand_dir, RUN_PREFIX + "_" + str(max_num + 1))
    os.makedirs(run_dir, exist_ok=True)

    return run_dir, os.path.join(run_dir, (MODEL_NAME + MODEL_EXT))


def get_model_params_count(model):
    return sum(p.numel() for p in model.parameters())


def save_history_plot(history_table, index, title, x_label, y_label, path):
    history_dict = history_table.to_dict(orient='list')
    history_dict[index] = history_table.index.tolist()
    history_dict = {key: np.array(value) for key, value in history_dict.items()}

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    x = history_dict[index]

    for key, val in history_dict.items():
        temp_x = x
        if key == 'global_epoch':
            continue

        if np.isnan(val).any():
            non_nan_mask = ~np.isnan(val)
            temp_x = x[non_nan_mask]
            val = val[non_nan_mask]
        ax.plot(temp_x, val, label=key)
        ax.legend()

    fig.savefig(path)
    plt.close(fig)


# An included library with Python install.
def async_message_box(title, text, style):
    thread = threading.Thread(target=ctypes.windll.user32.MessageBoxW,
                              args=(0, text, title, style), daemon=True)
    thread.start()


def plot_overlay(item_wise, color, alpha=1.0):
    changes = np.diff(np.concatenate(([0], item_wise, [0])))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    for start, end in zip(starts, ends):
        plt.axvspan(start, end, color=color, alpha=alpha)


def plot_target_prediction(wave, noised_wave, target, pred, sample_rate, save_path=None):
    plt.figure(figsize=(20, 4))

    plot_overlay(target, 'blue')

    matching_inds = np.logical_and(target == pred, target == 1)
    matches = np.zeros_like(pred)
    matches[matching_inds] = 1

    plot_overlay(matches, 'limegreen')

    non_matching_inds = np.logical_and(target != pred, target == 0)
    non_matching = np.zeros_like(pred)
    non_matching[non_matching_inds] = 1

    plot_overlay(non_matching, 'red')
    plt.plot(noised_wave[0], color='gray')
    plt.plot(wave[0], color='black')

    plt.yticks([])
    xticks = plt.xticks()[0]
    xticks = xticks[np.logical_and(xticks >= 0, xticks <= wave.shape[-1])]
    plt.xticks(xticks, xticks / sample_rate)

    if save_path is None:
        plt.show()
        plt.close()
        return

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()