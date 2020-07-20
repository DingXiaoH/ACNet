import torch
from torch.nn.parameter import Parameter
import torch.nn.init as init

class ScaleLayer(torch.nn.Module):

    def __init__(self, num_features, use_bias=True):
        super(ScaleLayer, self).__init__()
        self.weight = Parameter(torch.Tensor(num_features))
        init.ones_(self.weight)
        self.num_features = num_features

        if use_bias:
            self.bias = Parameter(torch.Tensor(num_features))
            init.zeros_(self.bias)
        else:
            self.bias = None


    def forward(self, inputs):
        if self.bias is None:
            return inputs * self.weight.view(1, self.num_features, 1, 1)
        else:
            return inputs * self.weight.view(1, self.num_features, 1, 1) + self.bias