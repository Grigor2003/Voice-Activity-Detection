import torch

from other.utils import WaveToMFCCConverter

mfcc = WaveToMFCCConverter(100, win_length=400)
x = torch.randn(1, 799+1)
print(mfcc(x).shape)
print(x.size(-1) / mfcc.win_length)