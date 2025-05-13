import os
import random

import numpy as np
import ruamel.yaml
from ruamel.yaml import CommentedMap, CommentedSeq

from other.parsing.train_args_helper import *
from other.models.models_handler import MODELS_COUNT, NAMES
from other.parsing.train_args_helper import SynthArgs

y_path = 'configs/train.yaml'

yaml = ruamel.yaml.YAML(typ='rt')
# Load YAML file
with open(y_path) as f:
    ydict = yaml.load(f)

run_desc = ydict['run_description']

# Data section
root = is_type_of(ydict['data']['root'], req=False)
seed = is_type_of(ydict['data']['seed'], int, req=False)
if seed is None:
    seed = random.randint(0, 2 ** 32 - 1)

clean_audios_path = is_type_of(ydict['data']['clean'], req=True)
clean_labels_path = is_type_of(ydict['data']['labels'], req=True)

# Synthetic data
empty_batches = is_range(ydict['data']['empty_batches'], 1, 2 ** 16, int, req=False)
synth_args = SynthArgs(ydict['data']['synthetic'])

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
    create_new_model = str(create_new_model_lit).lower() in ['true', '1', 't', 'y', 'yes', '+']
weights_load_from = is_type_of(ydict['model']['weights'], req=False)

saves_count = is_range(ydict['model']['saves_count'], 0, 100, int)

# Augmentation Noises
noise_args = NoiseArgs(ydict['augmentation']['noises'])

# Augmentation Impulses
impulse_args = ImpulseArgs(ydict['augmentation']['impulses'])

# Root fix
if root is not None:
    clean_audios_path = os.path.join(root, clean_audios_path)
    clean_labels_path = os.path.join(root, clean_labels_path)

    if synth_args.labels is not None:
        synth_args.labels_path = os.path.join(root, synth_args.labels_path)
        synth_args.dir = os.path.join(root, synth_args.dir)

    for d in noise_args.datas:
        d.data_dir = os.path.join(root, d.data_dir)
    impulse_args.mic_ir_dir = os.path.join(root, impulse_args.mic_ir_dir)

# Train section
lr_pow = is_range(ydict['train']['lr'], -100, 100)
lr = 10 ** lr_pow
do_epoches = is_range(ydict['train']['epoch'], 0, 1000, int)
num_workers = is_range(ydict['train']['workers'], 0, 32, int)
batch_size = is_range(ydict['train']['batch'], 1, 2 ** 15, int)
max_batches = is_range(ydict['train']['max_batches'], 1, 2 ** 15, int, req=False)
accumulation_steps = is_range(ydict['train']['n_accum'], 1, 2 ** 15, int)

max_batches = float('inf') if max_batches is None else max_batches

if saves_count == 0:
    saves_count = do_epoches
elif saves_count > do_epoches:
    raise ValueError(f"Saves count must be less than epoches count to do: {do_epoches}")
save_frames = np.linspace(do_epoches / saves_count, do_epoches, saves_count, dtype=int)

synth_args.post_zero_count(batch_size)
synth_args.post_count(batch_size)

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
noise_args.val_min_noise_count = is_range(ydict['val']['min_noise_count'], 1, 2 ** 15, int)

# Verbose section
threshold = is_range(ydict['verbose']['threshold'], 0, 1)
plot = is_type_of(ydict['verbose']['plot'], bool)
print_mbox = is_type_of(ydict['verbose']['mbox'], bool)
print_val_results = is_type_of(ydict['verbose']['val_results'], bool)
n_examples = ydict['verbose']['n_examples']
val_examples = is_range(ydict['verbose']['val_examples'], 1, 100, int, req=False)

run_desc = str.format(run_desc, e=do_epoches, l=lr_pow)


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


def strip_comments(data):
    if isinstance(data, CommentedMap):
        new_data = {}
        for k, v in data.items():
            new_data[k] = strip_comments(v)
        return new_data
    elif isinstance(data, CommentedSeq):
        return [strip_comments(i) for i in data]
    else:
        return data
