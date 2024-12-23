import torch.nn as nn
import torch.nn.functional as F


class SimpleG(nn.Module):
    def __init__(self, input_dim, hidden_dim, gru_num_layers=1):
        super(SimpleG, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, gru_num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

        self.input_dim = input_dim

    def forward(self, x, padding_mask=None):
        out, _ = self.gru(x)
        out = self.fc(out)
        out = F.sigmoid(out)

        return out


class SimpleDGGD(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, gru1_num_layers=1,
                 gru2_num_layers=1):
        super(SimpleDGGD, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.gru1 = nn.GRU(hidden_dim1, hidden_dim2, gru1_num_layers, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim2, hidden_dim3, gru2_num_layers, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim3, hidden_dim4)
        self.fc3 = nn.Linear(hidden_dim4, 1)

        self.input_dim = input_dim

    def forward(self, x, padding_mask=None):
        out = self.fc1(x)
        out = F.relu(out)
        out, _ = self.gru1(out)
        out, _ = self.gru2(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.sigmoid(out)

        return out


class DGGD(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, num_layers=1, dropout_prob=0.5):
        super(DGGD, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.layernorm1 = nn.LayerNorm(hidden_dim1)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.gru1 = nn.GRU(hidden_dim1, hidden_dim2, num_layers, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim2, hidden_dim3, num_layers, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim3, hidden_dim4)
        self.layernorm2 = nn.LayerNorm(hidden_dim4)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(hidden_dim4, 1)

        self.input_dim = input_dim

    def forward(self, x, padding_mask=None):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.layernorm1(out)
        out = self.dropout1(out)
        out, _ = self.gru1(out)
        out, _ = self.gru2(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.layernorm2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = F.sigmoid(out)
        return out
