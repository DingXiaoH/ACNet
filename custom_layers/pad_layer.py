import torch.nn as nn
import torch.nn.functional as F

class PadLayer(nn.Module):

    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, pad):
        super(PadLayer, self).__init__()
        self.pad = pad

    def forward(self, input):
        F.pad(input, [self.pad] * 4)