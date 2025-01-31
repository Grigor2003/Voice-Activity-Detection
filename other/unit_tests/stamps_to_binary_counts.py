import numpy as np
import random

from sympy import true
from tqdm import tqdm


def random_stamps_list(s=100, ones=(100, 100000), zeros=(1, 20000)):
    stamps = []
    for i in range(s):
        if len(stamps) == 0:
            f = random.choice([random.randint(*zeros), 0])
        else:
            f = stamps[-1][-1] + np.random.randint(*zeros)
        t = f + np.random.randint(*ones)
        stamps.append((f, t))
    return stamps


def stamps_to_binary_counts(stamps, needed_len):
    e_ = 0
    binary = []
    for s, e in stamps:
        binary.append(s - e_)
        binary.append(e - s)
        e_ = e
    if needed_len is not None and len(binary) < needed_len:
        binary.append(needed_len - len(binary))
    return binary


def binary_counts_to_binary(binary_counts):
    binary = []
    for i, counts in enumerate(binary_counts):
        add = i % 2
        binary.extend([add] * counts)
    return binary


def binary_counts_to_windows_alt(binary_counts, w):
    q = w // 2
    binary = binary_counts_to_binary(binary_counts)
    l = len(binary)
    counts = []
    frames = l // q
    for i in range(frames):
        counts.append(sum(binary[q * i:q * (i + 1)]))

    return counts


def binary_counts_to_windows(binary, w):
    q = w // 2
    s = 0
    curr = 0
    mek = 0
    ones = []
    while curr < len(binary):
        s += binary[curr]
        if s >= q:
            ones.append(mek)
            ones.extend((s // q - 1) * [0])
            s %= q
            mek = 0

        curr += 1
        s += binary[curr]
        if s >= q:
            ones.append(mek + q + binary[curr] - s)
            ones.extend((s // q - 1) * [q])
            s %= q
            mek = s
        else:
            mek += binary[curr]

        curr += 1
    # print(ones)
    return ones


def binary_counts_to_windows_np(binary, w, total=None):
    q = w // 2
    s, mek = 0, 0
    if total is None:
        total = sum(binary)
    count = total // q - 1
    ones = np.zeros(count, dtype=int)

    count_of_zeros = True
    i = 0
    for val in binary:
        s += val
        if count_of_zeros:
            if s >= q:
                if i == 0:
                    ones[i] += mek
                ones[i - 1:i + 1] += mek
                skip, s = divmod(s, q)
                i += skip

                mek = 0
        else:
            if s >= q:
                if i == 0:
                    ones[i] += mek + q + val - s
                ones[i - 1:i + 1] += mek + q + val - s
                skip, s = divmod(s, q)
                ones[i:i + skip - 1] += q
                ones[i + 1:i + skip] += q
                i += skip

                mek = s
            else:
                mek += val
        count_of_zeros = not count_of_zeros

    return ones


def binary_counts_to_windows_ai(binary, w):
    q = w // 2
    s, mek = 0, 0
    ones = []

    count_of_zeros = True
    append_ones = ones.append
    extend_ones = ones.extend

    for curr, val in enumerate(binary):
        s += val
        if count_of_zeros:
            if s >= q:
                append_ones(mek)
                extend_ones([0] * ((s // q) - 1))
                s %= q
                mek = 0
        else:
            if s >= q:
                append_ones(mek + q + val - s)
                extend_ones([q] * ((s // q) - 1))
                s %= q
                mek = s
            else:
                mek += val
        count_of_zeros = not count_of_zeros

    return ones


import time
from tqdm import tqdm

window = 20
errors = 0
attempts = 100

times_windows = []
times_windows_np = []
times_windows_ai = []
times_windows_alt = []
times_windows_numpy_sum = []
times_windows_lists_sum = []

for _ in tqdm(range(attempts), disable=0):
    stamps = random_stamps_list(s=200, ones=(1, 10000), zeros=(1, 5000))

    binary_counts = stamps_to_binary_counts(stamps, 10)
    total = sum(binary_counts)

    t1 = time.perf_counter()
    ones_alt = binary_counts_to_windows_alt(binary_counts, window)
    ones_alt = np.array(ones_alt)
    ones_alt = ones_alt[:-1] + ones_alt[1:]
    t2 = time.perf_counter()
    times_windows_alt.append(t2 - t1)

    ###########################

    t1 = time.perf_counter()
    ones_np = binary_counts_to_windows_np(binary_counts, window)
    t2 = time.perf_counter()
    times_windows_np.append(t2 - t1)

    # t1 = time.perf_counter()
    # ones_ai = binary_counts_to_windows_ai(binary_counts, window)
    # ones_ai = np.array(ones_ai)
    # ones_ai = ones_ai[:-1] + ones_ai[1:]
    # t2 = time.perf_counter()
    # times_windows_ai.append(t2 - t1)
    #
    # t1 = time.perf_counter()
    # ones = binary_counts_to_windows(binary_counts, window)
    # ones = np.array(ones)
    # ones = ones[:-1] + ones[1:]
    # t2 = time.perf_counter()
    # times_windows.append(t2 - t1)

    test = ones_np

    t1 = time.perf_counter()
    # test = test[:-1] + test[1:]
    t2 = time.perf_counter()
    times_windows_numpy_sum.append(t2 - t1)

    e = np.sum(np.abs(test - ones_alt))
    if e > 0:
        print(binary_counts)
        print(*np.argwhere(test != ones_alt))
        print(*test)
        print(*ones_alt)
    errors += e

# Compute and print mean execution times
print(f"Mean Execution Times:")
# print(f"binary_counts_to_windows       : {1000 * np.mean(times_windows):.6f}ms")
# print(f"binary_counts_to_windows_ai    : {1000 * np.mean(times_windows_ai):.6f}ms")
print(f"binary_counts_to_windows_np    : {1000 * np.mean(times_windows_np):.6f}ms")
print(f"binary_counts_to_windows_alt   : {1000 * np.mean(times_windows_alt):.6f}ms")
print()
print(f"times_windows_numpy_sum        : {1000 * np.mean(times_windows_numpy_sum):.6f}ms")
# print(f"times_windows_lists_sum        : {1000 * np.mean(times_windows_lists_sum):.6f}ms")

print(f"Error rate : {errors / attempts * 100:.2f}%")
