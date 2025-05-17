from other.models.combined import BattleVAD
from other.models.convolutional_models import ConvModel, EfficientModel
from other.models.custom_espanet import CustomESPANet
from other.models.gru_models import DCGCGCD_13_7, DCGCGCD_2, DGCGCD_13_7, DGCGD, DGCGCGD_13_7, DGGD
from other.models.attention_models import AttentionModel, WhisperLikeModel

whisper_like = lambda: WhisperLikeModel(input_dim=64)
attention = lambda: AttentionModel(input_dim=64, attention_dim=128, hidden_dim2=64, hidden_dim3=32, hidden_dim4=16,
                                   num_heads=4, dropout_prob=0.2)
# simple_gru = lambda: SimpleG(input_dim=64, hidden_dim=48)
# simple_gru_with_denses = lambda: SimpleDGGD(input_dim=64, hidden_dim1=48, hidden_dim2=32, hidden_dim3=16, hidden_dim4=8)
gruconv_with_denses = lambda: DGCGD(input_dim=64, hidden_dim1=48, hidden_dim2=32, hidden_dim3=16, hidden_dim4=8,
                               dropout_prob=0.2)
gruconv_with_denses_bigger_x2 = lambda: DGCGCGD_13_7(input_dim=48)
gru2conv2_with_denses_bigger_x2 = lambda: DGCGCD_13_7(input_dim=48)
convgru_with_denses_bigger_x2 = lambda: DCGCGCD_13_7(input_dim=48)

gru_with_denses = lambda : DGGD(48, 64, 64, 32, 32)
dcgcgcd_2 = lambda : DCGCGCD_2(48, 64, 64, 32, 32)

bottleneck = lambda: EfficientModel(64)

battle_vad = lambda: BattleVAD(48)

custom_espanet = lambda: CustomESPANet(48)
conv_model = lambda: ConvModel(48)

MODELS = {
    "WhisperLike_64": whisper_like,
    "Attention_64": attention,
    # "SimpleG_64": simple_gru,
    # "SimpleDGGD_64": simple_gru_with_denses,
    # "DGGD_64": gru_with_denses,
    # "DNGGND_7": gru_with_denses_and_norms,
    "DGCGCGD_13_7": gruconv_with_denses_bigger_x2,
    "DGCGCD_13_7": gru2conv2_with_denses_bigger_x2,
    'DCGCGCD_13_7': convgru_with_denses_bigger_x2,
    "DGCGD_7": gruconv_with_denses,
    "DGGD": gru_with_denses,
    "Bottleneck": bottleneck,
    "BattleVAD": battle_vad,
    "CustomESPANet": custom_espanet,
    "ConvModel": conv_model,
    "DCGCGCD_2": dcgcgcd_2,
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
