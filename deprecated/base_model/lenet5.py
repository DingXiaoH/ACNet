import torch.nn as nn
from builder import ConvBuilder

LENET5_DEPS = [20, 50, 500]

class LeNet5(nn.Module):

    def __init__(self, builder:ConvBuilder, deps):
        super(LeNet5, self).__init__()
        self.bd = builder
        stem = builder.Sequential()
        stem.add_module('conv1', builder.Conv2d(in_channels=1, out_channels=LENET5_DEPS[0], kernel_size=5, bias=True))
        stem.add_module('maxpool1', builder.Maxpool2d(kernel_size=2))
        stem.add_module('conv2', builder.Conv2d(in_channels=LENET5_DEPS[0], out_channels=LENET5_DEPS[1], kernel_size=5, bias=True))
        stem.add_module('maxpool2', builder.Maxpool2d(kernel_size=2))
        self.stem = stem
        self.flatten = builder.Flatten()
        self.linear1 = builder.Linear(in_features=LENET5_DEPS[1] * 16, out_features=LENET5_DEPS[2])
        self.relu1 = builder.ReLU()
        self.linear2 = builder.Linear(in_features=LENET5_DEPS[2], out_features=10)

    def forward(self, x):
        out = self.stem(x)
        # print(out.size())
        out = self.flatten(out)
        out = self.linear1(out)
        out = self.relu1(out)
        out = self.linear2(out)
        return out


def create_lenet5(cfg, builder):
    return LeNet5(builder=builder, deps=cfg.deps)
