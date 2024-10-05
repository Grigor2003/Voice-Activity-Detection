import numpy as np
import yaml
from other.parsing_utils import *
from models_handler import MODELS_COUNT, NAMES

# Load YAML file
with open('configs/train.yaml', 'r') as file:
    ydict = yaml.safe_load(file)

# Data section
clean_audios_path = is_type_of(ydict['data']['clean'])
clean_labels_path = is_type_of(ydict['data']['labels'])
noise_data_path = is_type_of(ydict['data']['noise'])

# Model section
model_id = with_range(ydict['model']['id'], 0, MODELS_COUNT, int, req=False)
model_name = is_type_of(ydict['model']['name'], req=False)
if model_id is not None:
    model_name = NAMES[model_id]
elif model_name is not None:
    if model_name not in NAMES:
        raise ValueError(f"Model name must be one of: {NAMES}")
else:
    raise ValueError(f"Model name or id has to be declared")


load_from = is_type_of(ydict['model']['weights'], req=False)
load_last = is_type_of(ydict['model']['use_last'], bool)

# Noise section
epoch_noise_count = with_range(ydict['noise']['pool'], 0, 5000, int)
snr = with_range(ydict['noise']['snr'], -20, 20)
augmentation_params = {
    "noise_count": with_range(ydict['noise']['count'], 0, 10, int),
    "noise_duration_range": parse_range(ydict['noise']['duration'], [0, 60], [0, 60]),
    "snr_db": snr
}

# Train section
lr = with_range(ydict['train']['lr'], -10, 10)
do_epoches = with_range(ydict['train']['epoch'], 0, 1000, int)
num_workers = with_range(ydict['train']['workers'], 0, 32, int)
batch_size = with_range(ydict['train']['batch'], 1, 2 ** 15, int)

# Result section
saves_count = with_range(ydict['result']['saves_count'], 0, 100, int)
if saves_count == 0:
    saves_count = do_epoches
elif saves_count > do_epoches:
    raise ValueError(f"Saves count must be less than epoches count to do: {do_epoches}")
save_frames = np.linspace(do_epoches / saves_count, do_epoches, saves_count, dtype=int)

train_res_dir = is_type_of(ydict['result']['directory'])

# Validation section
train_ratio = 1 - with_range(ydict['val']['ratio'])
val_every = with_range(ydict['val']['every'], 0, 1000, int)
val_num_workers = with_range(ydict['val']['workers'], 0, 32, int)
val_batch_size = with_range(ydict['val']['batch'], 1, 2 ** 15, int)

val_params = augmentation_params.copy()
del val_params["snr_db"]
val_snrs = [None, 10, 5, 0]

if snr in val_snrs:
    val_snrs.remove(snr)
val_snrs.insert(0, snr)

# Verbose section
threshold = with_range(ydict['verbose']['threshold'])
plot = is_type_of(ydict['verbose']['plot'], bool)
print_level = with_range(ydict['verbose']['print'], 0, 2, int)
