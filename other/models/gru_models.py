import torch
import torch.nn as nn


class DGCGD_Accent(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, num_layers=1, dropout_prob=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.activation1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)

        self.gru1 = nn.GRU(hidden_dim1, hidden_dim2, num_layers, batch_first=True)
        self.conv2d1 = nn.Conv2d(1, 1, kernel_size=(1, 7), padding='same')
        self.gru2 = nn.GRU(hidden_dim2, hidden_dim3, num_layers, batch_first=True)

        self.fc2 = nn.Linear(hidden_dim3, hidden_dim4)
        self.activation2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)

        self.accent_fc1 = nn.Linear(hidden_dim4, 5)
        self.accent_activation1 = nn.Sigmoid()

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

        out = self.accent_fc1(out)
        out = self.accent_activation1(out)
        out = torch.sum(out, 1)
        return out
