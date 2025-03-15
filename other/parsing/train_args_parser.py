import random

import numpy as np
import ruamel.yaml

from other.parsing.parsing_utils import *
from other.models.models_handler import MODELS_COUNT, NAMES

y_path = 'configs/train.yaml'

yaml = ruamel.yaml.YAML(typ='rt')
# Load YAML file
with open(y_path) as f:
    ydict = yaml.load(f)

# Data section
seed = is_type_of(ydict['data']['seed'], int, req=False)
if seed is None:
    seed = random.randint(0, 2 ** 32 - 1)
clean_audios_path = is_type_of(ydict['data']['clean'])
clean_labels_path = is_type_of(ydict['data']['labels'])
noise_data_dir = is_type_of(ydict['data']['noise'])

# Impulses
mic_ir_dir = is_type_of(ydict['impulses']['mic_ir_dir'])
mic_ir_prob = is_range(ydict['impulses']['mic_ir_prob'], 0, 1)

# Model section
model_id = is_range(ydict['model']['id'], 0, MODELS_COUNT, int, req=False)
model_name = is_type_of(ydict['model']['name'], req=False)
if model_id is not None:
    model_name = NAMES[model_id]
elif model_name is not None:
    if model_name not in NAMES:
        raise ValueError(f"Model name must be one of: {NAMES}")
else:
    raise ValueError(f"Model name or id has to be declared")

create_new_model = is_type_of(ydict['model']['create_new_model'], bool, req=False)
weights_load_from = is_type_of(ydict['model']['weights'], req=False)

saves_count = is_range(ydict['model']['saves_count'], 0, 100, int)

# Noise section
epoch_noise_count = is_range(ydict['noise']['pool'], 0, 5000, int)
aug_params = {
    "noise_count": is_range(ydict['noise']['count'], 0, 10, int),
    "noise_duration_range": parse_range(ydict['noise']['duration'], [0, 60], [0, 60]),
    "impulse_mic_prob": mic_ir_prob,
}
snr_dict = parse_numeric_dict(ydict['noise']['snr_dict'],
                              1, 100,
                              [-25, 25, True, False],
                              [0, 2 ** 16, True, True])
zero_rate = is_type_of(ydict['noise']['zero_rate'], (int, float))

# Train section
lr = 10 ** is_range(ydict['train']['lr'], -100, 100)
do_epoches = is_range(ydict['train']['epoch'], 0, 1000, int)
num_workers = is_range(ydict['train']['workers'], 0, 32, int)
batch_size = is_range(ydict['train']['batch'], 1, 2 ** 15, int)
accumulation_steps = is_range(ydict['train']['n_accum'], 1, 2 ** 15, int)

if saves_count == 0:
    saves_count = do_epoches
elif saves_count > do_epoches:
    raise ValueError(f"Saves count must be less than epoches count to do: {do_epoches}")
save_frames = np.linspace(do_epoches / saves_count, do_epoches, saves_count, dtype=int)

zero_count = None
if zero_rate < 0:
    zero_count = int(-zero_rate)
elif zero_rate > 0:
    zero_count = int(zero_rate * batch_size)
default_win_length = is_range(ydict['train']['win_length'], 1, 2 ** 15, int)

# Validation section
train_ratio = 1 - is_range(ydict['val']['ratio'], 0, 1)
val_every = is_range(ydict['val']['every'], 0, 1000, int)
if val_every > do_epoches:
    val_every = 0
val_num_workers = is_range(ydict['val']['workers'], 0, 32, int, req=False)
if val_num_workers is None:
    val_num_workers = num_workers
val_batch_size = is_range(ydict['val']['mini_batch'], 1, 2 ** 15, int)
val_snrs_list = parse_numeric_list(ydict['val']['snr_list'], 1, 100, -25, 25, int, False)

# Verbose section
threshold = is_range(ydict['verbose']['threshold'], 0, 1)
plot = is_type_of(ydict['verbose']['plot'], bool)
print_mbox = is_type_of(ydict['verbose']['mbox'], bool)
print_val_results = is_type_of(ydict['verbose']['val_results'], bool)
n_examples = is_range(ydict['verbose']['n_examples'], 0, 1000, int)


def model_has_been_saved():
    global ydict

    if weights_load_from is None and not create_new_model:
        return

    ydict['model']['weights'] = None
    ydict['model']['create_new_model'] = None

    try:
        with open(y_path, 'w') as f:
            yaml.dump(ydict, f)
    except IOError:
        print("WARNING: Couldn't change yaml file content")
