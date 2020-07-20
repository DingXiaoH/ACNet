import torch.nn as nn
from builder import ConvBuilder


class LeNet5BN(nn.Module):

    def __init__(self, builder:ConvBuilder, deps):
        super(LeNet5BN, self).__init__()
        self.bd = builder
        stem = builder.Sequential()
        stem.add_module('conv1', builder.Conv2dBNReLU(in_channels=1, out_channels=deps[0], kernel_size=5))
        stem.add_module('maxpool1', builder.Maxpool2d(kernel_size=2))
        stem.add_module('conv2', builder.Conv2dBNReLU(in_channels=deps[0], out_channels=deps[1], kernel_size=5))
        stem.add_module('maxpool2', builder.Maxpool2d(kernel_size=2))
        self.stem = stem
        self.flatten = builder.Flatten()
        self.linear1 = builder.IntermediateLinear(in_features=deps[1] * 16, out_features=500)
        self.relu1 = builder.ReLU()
        self.linear2 = builder.Linear(in_features=500, out_features=10)

    def forward(self, x):
        out = self.stem(x)
        out = self.flatten(out)
        out = self.linear1(out)
        out = self.relu1(out)
        out = self.linear2(out)
        return out


def create_lenet5bn(cfg, builder):
    return LeNet5BN(builder=builder, deps=cfg.deps)
