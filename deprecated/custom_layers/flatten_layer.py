import torch.nn as nn

class FlattenLayer(nn.Module):

    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, inputs):
        return inputs.view(inputs.size(0), -1)
