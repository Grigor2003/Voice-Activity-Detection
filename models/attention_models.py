import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionModel(nn.Module):
    def __init__(self, input_dim, attention_dim, hidden_dim2, hidden_dim3, hidden_dim4, num_heads=4, dropout_prob=0.5):
        super(AttentionModel, self).__init__()
        self.input_dim = input_dim

        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, attention_dim)
        self.layernorm1 = nn.LayerNorm(attention_dim)
        self.dropout1 = nn.Dropout(dropout_prob)

        # First attention layer
        self.attention1 = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, batch_first=True)
        self.layernorm2 = nn.LayerNorm(attention_dim)

        # Second attention layer
        self.attention2 = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, batch_first=True)
        self.layernorm3 = nn.LayerNorm(attention_dim)

        # Fully connected layers after attention
        self.fc2 = nn.Linear(attention_dim, hidden_dim2)
        self.layernorm4 = nn.LayerNorm(hidden_dim2)
        self.dropout2 = nn.Dropout(dropout_prob)

        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.layernorm5 = nn.LayerNorm(hidden_dim3)
        self.dropout3 = nn.Dropout(dropout_prob)

        self.fc4 = nn.Linear(hidden_dim3, hidden_dim4)
        self.layernorm6 = nn.LayerNorm(hidden_dim4)
        self.dropout4 = nn.Dropout(dropout_prob)

        # Output layer
        self.fc_out = nn.Linear(hidden_dim4, 1)

    def forward(self, x, padding_mask=None):
        # Initial fully connected layer
        x = self.fc1(x)
        x = self.layernorm1(x)
        x = self.dropout1(x)

        # First attention layer with residual connection
        attn_output1, _ = self.attention1(x, x, x)
        x = x + attn_output1  # Residual connection
        x = self.layernorm2(x)

        # Second attention layer with residual connection
        attn_output2, _ = self.attention2(x, x, x)
        x = x + attn_output2  # Residual connection
        x = self.layernorm3(x)

        # Fully connected layers
        x = self.fc2(x)
        x = self.layernorm4(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.layernorm5(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        x = self.layernorm6(x)
        x = self.dropout4(x)

        # Final output layer
        x = self.fc_out(x)
        x = F.sigmoid(x)

        return x


class WhisperLikeModel(nn.Module):
    def __init__(self, input_dim, conv_dim=512, encoder_dim=512, num_heads=8, num_encoder_layers=6):
        super(WhisperLikeModel, self).__init__()

        self.input_dim = input_dim

        # Initial convolutional layers for audio feature extraction
        self.conv1 = nn.Conv1d(input_dim, conv_dim // 2, kernel_size=3, padding="same")
        self.conv2 = nn.Conv1d(conv_dim // 2, conv_dim, kernel_size=3, padding="same")
        self.conv3 = nn.Conv1d(conv_dim, conv_dim // 2, kernel_size=3, padding="same")
        self.conv4 = nn.Conv1d(conv_dim // 2, input_dim, kernel_size=3, padding="same")

        self.embedding = nn.Linear(input_dim, encoder_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=encoder_dim, nhead=num_heads,
                                                   dim_feedforward=encoder_dim * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers, enable_nested_tensor=False)

        # Projection to vocabulary size for output
        self.fc_out = nn.Linear(encoder_dim, 1)

    def forward(self, x, padding_mask=None):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.transpose(1, 2)

        # Encoder pass
        x = self.embedding(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        x = self.fc_out(x)
        x = F.sigmoid(x)

        return x
