import numpy as np
import yaml
from other.models.models_handler import MODELS_COUNT, NAMES

# Load YAML file
with open('configs/metrix.yaml', 'r') as file:
    ydict = yaml.safe_load(file)

# Data section
clean_audios_path = is_type_of(ydict['data']['clean'])
clean_labels_path = is_type_of(ydict['data']['labels'])
noise_data_path = is_type_of(ydict['data']['noise'])
enot_data_path = is_type_of(ydict['data']['enot_data'], req=False)

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

load_from = is_type_of(ydict['model']['weights'], req=True)

# Noise section
epoch_noise_count = is_range(ydict['noise']['pool'], 0, 5000, int)
snr = {is_range(ydict['noise']['snr'], -20, 20, int): 1}
augmentation_params = {
    "noise_count": is_range(ydict['noise']['count'], 0, 10, int),
    "noise_duration_range": parse_range(ydict['noise']['duration'], [0, 60], [0, 60]),
}

# Train section
num_workers = is_range(ydict['train']['workers'], 0, 32, int)
batch_size = is_range(ydict['train']['batch'], 1, 2 ** 15, int)

# ROC section
thresholds = np.linspace(*parse_linspace(ydict['ROC']['thresholds'], [0, 1], [0, 1], [1, 1000, int]))
