import torch.nn as nn
from builder import ConvBuilder
from constants import CFQK_ORIGIN_DEPS


class CFQKBN(nn.Module):

    def __init__(self, num_classes, builder:ConvBuilder, deps=None):
        super(CFQKBN, self).__init__()
        if deps is None:
            deps = CFQK_ORIGIN_DEPS
        self.bd = builder
        self.conv1 = self.bd.Conv2dBNReLU(in_channels=3, out_channels=deps[0], kernel_size=5, stride=1, padding=2)
        self.conv2 = self.bd.Conv2dBNReLU(in_channels=deps[0], out_channels=deps[1], kernel_size=5, stride=1, padding=2)
        self.conv3 = self.bd.Conv2dBNReLU(in_channels=deps[1], out_channels=deps[2], kernel_size=5, stride=1, padding=2)
        self.fc1 = self.bd.Linear(in_features=3*3*deps[2], out_features=64)
        self.fc2 = self.bd.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)   #   32
        x = self.bd.max_pool2d(x, kernel_size=3, stride=2, padding=0)   #15
        x = self.conv2(x)
        x = self.bd.avg_pool2d(x, kernel_size=3, stride=2, padding=0)   #7
        x = self.conv3(x)
        x = self.bd.avg_pool2d(x, kernel_size=3, stride=2, padding=0)   #3
        x = self.bd.flatten(x)
        x = self.fc1(x)
        x = self.bd.relu(x)
        x = self.fc2(x)
        return x

def create_CFQKBNC(cfg, builder):
    return CFQKBN(num_classes=10, builder=builder, deps=cfg.deps)

def create_CFQKBNH(cfg, builder):
    return CFQKBN(num_classes=100, builder=builder, deps=cfg.deps)