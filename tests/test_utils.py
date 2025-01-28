import torch
import torchaudio


x, sr = torchaudio.load(r"C:\Users\gg\Projects\Voice-Activity-Detection\data\103-1240-0000.flac")

print(torch.sum(x**2))
print(torch.max(abs(x)))