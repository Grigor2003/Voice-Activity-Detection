import numpy as np
import torch


class AudioBinaryLabel:

    def clean(self):
        self.binary = None
        self.one_stamps = None
        self.length = None

    def __init__(self):
        self.binary = None
        self.one_stamps = None
        self.length = None

    @staticmethod
    def from_one_stamps(one_stamps, length, to=None):
        if to is None:
            to = AudioBinaryLabel()
        to.clean()
        to.one_stamps = one_stamps
        to.length = length
        return to

    @staticmethod
    def from_binary(binary, to=None):
        if to is None:
            to = AudioBinaryLabel()
        to.clean()
        to.binary = binary
        return to

    # Get or compute
    def binary_goc(self):
        if self.binary is None:
            self.binary = stamps_to_binary_counts(self.one_stamps, self.length)
        return self.binary


def stamps_to_binary_counts(stamps, target_len):
    e_ = 0
    summa = 0
    binary = []
    for s, e in stamps:
        binary.append(s - e_)
        binary.append(e - s)
        summa += e - e_
        e_ = e
    if target_len is not None and summa < target_len:
        binary.append(target_len - summa)
    return binary


def binary_counts_to_windows_np(binary, window, total=None):
    # Math says it must work
    half_window = window // 2
    current_diff, ones_count = 0, 0
    if total is None:
        total = sum(binary)
    count = total // half_window - 1
    ones = np.zeros(count, dtype=int)

    stepping_on_zeros = True
    i = 0
    for val in binary:
        current_diff += val
        if stepping_on_zeros:
            if current_diff >= half_window:
                if i == 0:
                    ones[i] += ones_count
                ones[i - 1:i + 1] += ones_count
                skip, current_diff = divmod(current_diff, half_window)
                i += skip

                ones_count = 0
        else:
            if current_diff >= half_window:
                if i == 0:
                    ones[i] += ones_count + half_window + val - current_diff
                ones[i - 1:i + 1] += ones_count + half_window + val - current_diff
                skip, current_diff = divmod(current_diff, half_window)
                ones[i:i + skip - 1] += half_window
                ones[i + 1:i + skip] += half_window
                i += skip

                ones_count = current_diff
            else:
                ones_count += val
        stepping_on_zeros = not stepping_on_zeros

    return ones


def top_k_indices(lst, k):
    return [idx for idx, _ in sorted(enumerate(lst), key=lambda x: x[1], reverse=True)[:k]]


def balance_regions(wave, counts, k=3):
    """
    balances ones and zeros counts by evenly distributing the deficit of zeros in the largest k zero regions
    """
    zeros = counts[::2]
    ones = counts[1::2]
    zeros_sum = sum(zeros)
    ones_sum = sum(ones)
    if zeros_sum >= ones_sum:
        return wave, counts

    diff = ones_sum - zeros_sum
    largest_zero_regions_indices = top_k_indices(zeros, k)
    add = diff // k
    for idx in largest_zero_regions_indices:
        count = zeros[idx]
        counts_before = sum(counts[:idx * 2])
        fill_coord = counts_before + count // 2
        first_half = wave[:, :fill_coord]
        second_half = wave[:, fill_coord:]
        wave = torch.cat([first_half, torch.zeros(wave.size(0), add), second_half], dim=-1)
        zeros[idx] += add
        counts[idx * 2] += add

    return wave, counts
