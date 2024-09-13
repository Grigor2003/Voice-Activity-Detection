from models.gru_model import SimpleG, SimpleDGGD, DGGD

simple_gru = SimpleG(input_dim=64, hidden_dim=48)
simple_gru_with_denses = SimpleDGGD(input_dim=64, hidden_dim1=48, hidden_dim2=32, hidden_dim3=16, hidden_dim4=8)
gru_with_denses = DGGD(input_dim=64, hidden_dim1=48, hidden_dim2=32, hidden_dim3=16, hidden_dim4=8, dropout_prob=0.2)

MODELS = {
    "SimpleG_64_48": simple_gru,
    "SimpleDGGD_64_48_32_16_8": simple_gru_with_denses,
    "DGGD_64_48_32_16_8": gru_with_denses,
    "Other": None
}

NAMES = [*MODELS.keys()]
MODELS_COUNT = len(NAMES)

if MODELS_COUNT == 0:
    raise Exception("You must specify at least one model")
