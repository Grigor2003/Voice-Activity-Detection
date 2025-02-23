import os
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from tabulate import tabulate
import ctypes
import threading


RES_PREFIX = "res"
DATE_FORMAT = "%Y-%m-%d"
MODEL_NAME = "weights.pt"
EXAMPLE_FOLDER = "examples"
RES_FOLDER = "train_results_accent"


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


def print_as_table(dataframe):
    if len(dataframe) > 4:
        print(tabulate(dataframe.iloc[[0, -3, -2, -1], :].T.fillna("---"), headers='keys'))
    else:
        print(tabulate(dataframe.T.fillna("---"), headers='keys'))


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
        date_objects = [datetime.strptime(date, DATE_FORMAT)
                        for date in os.listdir(model_trains_tree_dir)
                        if len(os.listdir(os.path.join(model_trains_tree_dir, date))) != 0]
        if len(date_objects) != 0:
            max_num = 0
            day_dir = os.path.join(model_trains_tree_dir, max(date_objects).strftime(DATE_FORMAT))
            for name in os.listdir(day_dir):
                st, num = name.split("_")
                folder_path = os.path.join(day_dir, name)
                if max_num <= int(num) and MODEL_NAME in os.listdir(folder_path):
                    max_num = int(num)
                    res_dir = folder_path

    if res_dir is None:
        return None, None
    else:
        return res_dir, os.path.join(res_dir, MODEL_NAME)


def create_new_model_trains_dir(model_name):
    model_trains_tree_dir = os.path.join(RES_FOLDER, model_name)

    day_dir = os.path.join(model_trains_tree_dir, datetime.now().strftime(DATE_FORMAT))
    os.makedirs(day_dir, exist_ok=True)
    max_num = 0
    for name in os.listdir(day_dir):
        _, num = name.split("_")
        max_num = max(int(num), max_num)

    dir = os.path.join(day_dir, RES_PREFIX + "_" + str(max_num + 1))
    os.makedirs(dir, exist_ok=True)

    return dir, os.path.join(dir, MODEL_NAME)


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
