import torch
import torch.nn as nn

class MaxLayer(nn.Module):

    def __init__(self):
        super(MaxLayer, self).__init__()

    def forward(self, a, b):
        return torch.max(a, b)
