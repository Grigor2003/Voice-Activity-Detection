import torch.nn as nn


class SimpleG(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(SimpleG, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out)
        return out
