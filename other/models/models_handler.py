from other.models.combined import BattleVAD
from other.models.convolutional_models import EfficientModel
from other.models.gru_models import DGCGD_7, DGCGCGD_13_7
from other.models.attention_models import AttentionModel, WhisperLikeModel

whisper_like = lambda: WhisperLikeModel(input_dim=64)
attention = lambda: AttentionModel(input_dim=64, attention_dim=128, hidden_dim2=64, hidden_dim3=32, hidden_dim4=16,
                                   num_heads=4, dropout_prob=0.2)
# simple_gru = lambda: SimpleG(input_dim=64, hidden_dim=48)
# simple_gru_with_denses = lambda: SimpleDGGD(input_dim=64, hidden_dim1=48, hidden_dim2=32, hidden_dim3=16, hidden_dim4=8)
gruconv_with_denses = lambda: DGCGD_7(input_dim=64, hidden_dim1=48, hidden_dim2=32, hidden_dim3=16, hidden_dim4=8,
                                      dropout_prob=0.2)
gruconv_with_denses_bigger_x2 = lambda: DGCGCGD_13_7(input_dim=64)

bottleneck = lambda: EfficientModel(64)

battle_vad = lambda: BattleVAD(64)

MODELS = {
    "WhisperLike_64": whisper_like,
    "Attention_64": attention,
    # "SimpleG_64": simple_gru,
    # "SimpleDGGD_64": simple_gru_with_denses,
    # "DGGD_64": gru_with_denses,
    # "DNGGND_7": gru_with_denses_and_norms,
    "DGCGCGD_13_7": gruconv_with_denses_bigger_x2,
    "DGCGD_7": gruconv_with_denses,
    "Bottleneck": bottleneck,
    "BattleVAD": battle_vad,
    "Other": None
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
