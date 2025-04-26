import torch
import torch.nn as nn


class DGCGCGD_13_7(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.activation1 = nn.Tanh()
        self.dropout2 = nn.Dropout(0.2)

        self.gru1 = nn.GRU(128, 64, 1, batch_first=True)
        self.conv2d1 = nn.Conv2d(1, 1, kernel_size=(13, 13), padding='same')
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.gru2 = nn.GRU(64, 32, 1, batch_first=True)
        self.conv2d2 = nn.Conv2d(1, 1, kernel_size=(7, 7), padding='same')
        self.bn2 = nn.BatchNorm1d(num_features=32)
        self.gru3 = nn.GRU(32, 16, 1, batch_first=True)

        self.fc2 = nn.Linear(16, 12)
        self.activation2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(12, 1)
        self.activation3 = nn.Sigmoid()

        self.input_dim = input_dim
        self.hidden_states = None

    def forward(self, x: torch.Tensor, padding_mask=None, hidden_state=None):
        batch_size, seq_length = x.size(0), x.size(1)
        out = self.fc1(x)
        out = self.activation1(out)
        # region GRU block
        if hidden_state is None:
            hidden_state = [None] * 3
        hiddens1, _ = self.gru1(out, hidden_state[0])
        conv_in = hiddens1.view(batch_size, 1, seq_length, hiddens1.size(-1))
        conv_out = self.conv2d1(conv_in)
        conv_out = conv_out.view(-1, seq_length, conv_out.size(-1)).transpose(1, 2)
        conv_out = self.bn1(conv_out).transpose(1, 2)

        hiddens2, _ = self.gru2(conv_out, hidden_state[1])
        conv_in = hiddens2.view(batch_size, -1, seq_length, hiddens2.size(-1))
        conv_out = self.conv2d2(conv_in)
        conv_out = conv_out.view(-1, seq_length, conv_out.size(-1)).transpose(1, 2)
        conv_out = self.bn2(conv_out).transpose(1, 2)

        hiddens3, _ = self.gru3(conv_out, hidden_state[2])
        self.hidden_states = [hiddens1, hiddens2, hiddens3]
        # endregion

        out = self.fc2(hiddens3)
        out = self.activation2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = self.activation3(out)
        return out


class DGCGD_7(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, num_layers=1, dropout_prob=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.activation1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)

        self.gru1 = nn.GRU(hidden_dim1, hidden_dim2, num_layers, batch_first=True)
        self.conv2d1 = nn.Conv2d(1, 1, kernel_size=(7, 7), padding='same')
        self.gru2 = nn.GRU(hidden_dim2, hidden_dim3, num_layers, batch_first=True)

        self.fc2 = nn.Linear(hidden_dim3, hidden_dim4)
        self.activation2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)

        self.fc3 = nn.Linear(hidden_dim4, 1)
        self.activation3 = nn.Sigmoid()

        self.input_dim = input_dim
        self.hidden_states = None

    def forward(self, x: torch.Tensor, padding_mask=None, hidden_state=None):
        out = self.fc1(x)
        out = self.activation1(out)
        out = self.dropout1(out)

        # region GRU block
        h1, h2 = None, None
        if hidden_state is not None:
            h1, h2 = hidden_state
        hiddens1, _ = self.gru1(out, h1)
        conv_out = self.conv2d1(hiddens1.unsqueeze(1)).squeeze(1)
        hiddens2, _ = self.gru2(conv_out, h2)
        self.hidden_states = [hiddens1, hiddens2]
        # endregion

        out = self.fc2(hiddens2)
        out = self.activation2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = self.activation3(out)
        return out
