import yaml
from other.parsing_utils import *
from models_handler import MODELS_COUNT, NAMES

# Load YAML file
with open('configs/train.yaml', 'r') as file:
    args = yaml.safe_load(file)

# Data section
clean = is_type_of(args['data']['clean'])
labels = is_type_of(args['data']['labels'])
noise = is_type_of(args['data']['noise'])

# Model section
use_last = is_type_of(args['model']['use_last'], bool)
model_id = with_range(args['model']['id'], 0, MODELS_COUNT, int)
model_name = is_type_of(args['model']['name'], req=False)
model_path = is_type_of(args['model']['path'], req=False)

# Noise section
noise_pool = with_range(args['noise']['pool'], 0, 5000, int)
noise_count = with_range(args['noise']['count'], 0, 10, int)
noise_duration = parse_list(args['noise']['duration'], [0, 60], [0, 60])
snr = with_range(args['noise']['snr'], -20, 20)

# Train section
lr = with_range(args['train']['lr'], -10, 10)
epoch = with_range(args['train']['epoch'], 0, 1000, int)
workers = with_range(args['train']['workers'], 0, 32, int)
batch = with_range(args['train']['batch'], 1, 2 ** 15, int)

# Result section
saves_count = with_range(args['result']['saves_count'], 1, 100, int)
train_res = is_type_of(args['result']['directory'])

# Validation section
val_ratio = with_range(args['val']['ratio'])
val_every = with_range(args['val']['every'], 0, 1000, int)
val_workers = with_range(args['val']['workers'], 0, 32, int)
val_batch = with_range(args['val']['batch'], 1, 2 ** 15, int)

# Verbose section
threshold = with_range(args['verbose']['threshold'])
no_plot = is_type_of(args['verbose']['no_plot'], bool)
print_level = with_range(args['verbose']['print'], 0, 2, int)
