import yaml
from other.parsing_utils import *
from models_handler import MODELS_COUNT, NAMES

# Load YAML file
with open('configs/metrix.yaml', 'r') as file:
    ydict = yaml.safe_load(file)

# Data section
clean_audios_path = is_type_of(ydict['data']['clean'])
clean_labels_path = is_type_of(ydict['data']['labels'])
noise_data_path = is_type_of(ydict['data']['noise'])
enot_data_path = is_type_of(ydict['data']['enot_data'], req=False)

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

# Noise section
epoch_noise_count = with_range(ydict['noise']['pool'], 0, 5000, int)
noise_count = with_range(ydict['noise']['count'], 0, 10, int)
noise_duration = parse_list(ydict['noise']['duration'], [0, 60], [0, 60])
snr = with_range(ydict['noise']['snr'], -20, 20)

# Train section
num_workers = with_range(ydict['train']['workers'], 0, 32, int)
batch_size = with_range(ydict['train']['batch'], 1, 2 ** 15, int)

# Verbose section
threshold = with_range(ydict['verbose']['threshold'])
no_plot = is_type_of(ydict['verbose']['no_plot'], bool)
print_level = with_range(ydict['verbose']['print'], 0, 2, int)
