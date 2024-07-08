from gru_model import SimpleG

simple_gru = SimpleG(input_dim=64, hidden_dim=48)

MODELS = {
    "SimpleG_64_48": simple_gru,
    "Other": None
}

NAMES = [*MODELS.keys()]
