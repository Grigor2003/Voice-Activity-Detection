import numpy as np
import random

from tqdm import tqdm


def one_stamping(source, step, window):
    ones_regions = []
    last_being = False
    for i, is_speech in enumerate(source):
        s = i * step
        e = s + window

        if last_being and is_speech:  # 1 -> 1
            ones_regions[-1] = e
        elif not last_being and is_speech:  # 0 -> 1
            if not ones_regions or ones_regions[-1] < s:
                ones_regions.extend([s, e])
            else:
                ones_regions[-1] = e
            last_being = True
        else:
            last_being = False

    return ones_regions


def item_wise_mask(source, step, window):
    mask = [0] * ((len(source) - 1) * step + window)
    mask = np.array(mask)

    for i, is_speech in enumerate(source):
        s = i * step
        e = s + window
        mask[s:e] = is_speech or mask[s:e]
    return mask


def stamp_to_mask(source):
    if len(source) == 0:
        return [0]
    mask = [0] * source[-1]
    mask = np.array(mask)
    for s, e in zip(source[::2], source[1::2]):
        mask[s:e] = 1

    return mask


step = 2
window = 100


x = lambda: random.randint(0, 1)
y = lambda: [x()] * (int(np.abs(np.random.randn() * 100)) + 1)
z = lambda: [y() for _ in range(random.randint(1, 100))]
a = lambda: [i for j in z() for i in j]

c = 0
empty_count = 0
for _ in tqdm(range(1000)):

    random_list = a()

    mask = item_wise_mask(random_list, step, window)
    stamps = one_stamping(random_list, step, window)
    recreated_mask = stamp_to_mask(stamps)
    empty_count += len(stamps) == 0

    if np.any(recreated_mask != mask[:len(recreated_mask)]):
        c += 1
        print('=' * 180)
        print(mask)
        print(recreated_mask)
        input("Press Enter to continue...")

print("Empty lists:", empty_count)
