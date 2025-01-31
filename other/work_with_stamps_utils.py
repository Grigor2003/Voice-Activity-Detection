import numpy as np


def stamps_to_binary_counts(stamps, target_len):
    e_ = 0
    binary = []
    for s, e in stamps:
        binary.append(s - e_)
        binary.append(e - s)
        e_ = e
    if target_len is not None and len(binary) < target_len:
        binary.append(target_len - len(binary))
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
