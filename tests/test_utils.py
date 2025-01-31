import torch
import torchaudio


x = torch.nn.Sequential(torch.nn.Linear(1, 1, 44100).to(torch.device('cpu')))
y = x.to(torch.device('cuda:0'))
print([*x.parameters()][0].device)