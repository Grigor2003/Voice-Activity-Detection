import random

import numpy as np
import ruamel.yaml

from other.parsing.parsing_utils import *
from other.parsing.train_args_helper import *
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

create_new_model_lit = is_type_of(ydict['model']['create_new_model'], req=False)
if create_new_model_lit is None:
    create_new_model = None
else:
    create_new_model = create_new_model_lit.lower() in ['true', '1', 't', 'y', 'yes', '+']
weights_load_from = is_type_of(ydict['model']['weights'], req=False)

saves_count = is_range(ydict['model']['saves_count'], 0, 100, int)

# Augmentation Noises
noise_args = NoiseArgs(ydict['augmentation']['noises'])

# Augmentation Impulses
impulse_args = ImpulseArgs(ydict['augmentation']['impulses'])

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

if noise_args.zero_rate < 0:
    noise_args.zero_count = int(-noise_args.zero_rate)
elif noise_args.zero_rate > 0:
    noise_args.zero_count = int(noise_args.zero_rate * batch_size)

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
    ydict['model']['create_new_model'] = ' '

    try:
        with open(y_path, 'w') as f:
            yaml.dump(ydict, f)
    except IOError:
        print("WARNING: Couldn't change yaml file content")
