from other.models.gru_models import DGCGD_Accent
from other.models.attention_models import AttentionModel, WhisperLikeModel


dgcgd_accent = lambda: DGCGD_Accent(input_dim=64, hidden_dim1=48, hidden_dim2=32, hidden_dim3=16, hidden_dim4=8,
                               dropout_prob=0.2)

MODELS = {
    "DGCGD_A_64": dgcgd_accent
}

NAMES = [*MODELS.keys()]
MODELS_COUNT = len(NAMES)

if MODELS_COUNT == 0:
    raise Exception("You must specify at least one model")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_vram_usage(model, include_gradients=True):
    # Calculate total number of parameters
    total_params = count_parameters(model)
    param_memory = total_params * 4  # 4 bytes per float32 parameter

    # If gradients are stored (common in training), double the memory usage
    if include_gradients:
        param_memory *= 2

    # Convert to (GB)
    return param_memory / (1024 ** 3)
