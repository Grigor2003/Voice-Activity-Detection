import torch
import torch.nn as nn


class SEWeight(nn.Module):
    def __init__(self, channels, reduction=2):
        super(SEWeight, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class PSA(nn.Module):
    def __init__(self, channel=48, kernel_sizes=[3, 5, 7], group_sizes=[2, 4, 8], reduction=8):
        super().__init__()
        self.channel = channel
        self.reduction = reduction
        self.split_count = len(kernel_sizes)

        # Split channels into n groups
        self.convs = nn.ModuleList()
        for i in range(self.split_count):
            self.convs.append(
                nn.Conv2d(channel//self.split_count, channel//self.split_count, kernel_size=kernel_sizes[i],
                          padding=kernel_sizes[i]//2, groups=group_sizes[i])
            )

        self.se_weights = SEWeight(channel, reduction)
        self.softmax = nn.Softmax(dim=1)
        self.split = Split(channel, self.split_count)

    def forward(self, x):
        # Split features into n groups
        x_split = self.split(x)

        # Multi-scale processing
        feats = [conv(x_split[i]) for i, conv in enumerate(self.convs)]
        feats = torch.cat(feats, dim=1)

        # Attention computation
        attn = self.se_weights(feats)
        attn = self.softmax(attn)

        # Feature aggregation
        return (feats * attn).sum(dim=1, keepdim=True)


class PSABlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.psa = PSA(out_channels, reduction=reduction)
        self.conv2 = nn.Conv2d(1, in_channels, 1)
        # self.batch_norm = nn.BatchNorm2d(in_channels)
        # self.dropout = nn.Dropout(0.2)

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.psa(x)
        x = self.conv2(x)
        # x = self.dropout(x) + inp
        # x = self.batch_norm(x)
        return x


class Split(nn.Module):
    def __init__(self, channel, split_count):
        super().__init__()
        self.split_channels = [channel//split_count]*split_count
        self.split_channels[0] += channel - sum(self.split_channels)

    def forward(self, x):
        return torch.split(x, self.split_channels, dim=1)


class CustomESPANet(nn.Module):

    def __init__(self, input_dim, channel_counts=[24, 48, 24], reduction=2):
        super().__init__()
        self.input_dim = input_dim
        self.psa_block1 = PSABlock(1, channel_counts[0], reduction=reduction)
        self.psa_block2 = PSABlock(1, channel_counts[1], reduction=reduction)
        self.psa_block3 = PSABlock(1, channel_counts[2], reduction=reduction)
        # self.conv1 = nn.Conv2d(channel_counts[2], 1, 1)
        self.fc1 = nn.Linear(input_dim, 64)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 4)

    def forward(self, inp, mask):
        x = self.psa_block1(inp.unsqueeze(1))
        x = self.psa_block2(x)
        x = self.psa_block3(x).squeeze(1)
        # x = self.conv1(x)
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.fc2(x)
        return x
