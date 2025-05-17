import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride):
        super(Bottleneck, self).__init__()
        mid_channels = in_channels * expansion
        self.use_residual = (stride == 1 and in_channels == out_channels)

        self.expand = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride,
                                   padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.project = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu6(self.bn1(self.expand(x)))
        out = F.relu6(self.bn2(self.depthwise(out)))
        out = self.bn3(self.project(out))
        return x + out if self.use_residual else out


class EfficientModel(nn.Module):
    def __init__(self, input_dim):
        self.input_dim = input_dim
        super(EfficientModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3,
                               stride=(1, 2), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # Define bottleneck layer settings: (in, out, expansion, stride)
        settings = [
            (32, 16, 1, (1, 1)),
            (16, 24, 6, (1, 2)),
            (24, 32, 6, (1, 2)),
            (32, 64, 6, (1, 2)),
            (64, 96, 6, (1, 1)),
        ]

        self.bottlenecks = nn.ModuleList()
        for in_c, out_c, t, s in settings:
            self.bottlenecks.append(Bottleneck(in_c, out_c, t, s))

        self.conv2 = nn.Conv2d(96, 1280, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)

        self.fc = nn.Linear(1280, 1)

    def forward(self, x, mask=None):
        x = x.unsqueeze(1)
        x = F.relu6(self.bn1(self.conv1(x)))

        for block in self.bottlenecks:
            x = block(x)

        x = F.relu6(self.bn2(self.conv2(x)))
        x = torch.mean(x, dim=-1, keepdim=True)
        x = x.squeeze(-1).transpose(1, 2)
        x = self.fc(x)
        x = F.sigmoid(x)
        return x

class Conv7_13Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_7 = nn.Conv2d(in_channels//2, out_channels//2, (7, 7), padding='same')
        self.conv_13 = nn.Conv2d(in_channels//2, out_channels//2, (13, 13), padding='same')
        self.activation = nn.ReLU()
        
    def forward(self, x):
        b, c, h, w = x.shape
        out_7 = self.conv_7(x[:,:c//2])
        out_13 = self.conv_7(x[:,c//2:])
        out = torch.cat([out_7, out_13], dim=1)
        return self.activation(out)

class ConvModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.expand = nn.Conv2d(1, 16, kernel_size=(1, 1))
        self.conv_block_1 = Conv7_13Block(16, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv_block_2 = Conv7_13Block(16, 32)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv_block_3 = Conv7_13Block(32, 16)
        self.bn3 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(input_dim, 64) #change to 16
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 4)
        
    def forward(self, x, mask=None):
        x = self.expand(x.unsqueeze(1))
        out = self.conv_block_1(x)
        out = self.bn1(out)
        out = self.conv_block_2(out)
        out = self.bn2(out)
        out = self.conv_block_3(out)
        out = self.bn3(out)
        out = out.mean(dim=-3)
        out = self.fc1(out.squeeze(1))
        out = self.activation1(out)
        self.prelast = out.detach()
        out = self.fc2(out)
        return out
        