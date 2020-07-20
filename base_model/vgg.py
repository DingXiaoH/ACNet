import torch.nn as nn
from builder import ConvBuilder
from constants import VGG_ORIGIN_DEPS

def _create_vgg_stem(builder, deps):
    sq = builder.Sequential()
    sq.add_module('conv1',
                  builder.Conv2dBNReLU(in_channels=3, out_channels=deps[0], kernel_size=3, stride=1, padding=1))
    sq.add_module('conv2',
                  builder.Conv2dBNReLU(in_channels=deps[0], out_channels=deps[1], kernel_size=3, stride=1, padding=1))
    sq.add_module('maxpool1', builder.Maxpool2d(kernel_size=2))
    sq.add_module('conv3',
                  builder.Conv2dBNReLU(in_channels=deps[1], out_channels=deps[2], kernel_size=3, stride=1, padding=1))
    sq.add_module('conv4',
                  builder.Conv2dBNReLU(in_channels=deps[2], out_channels=deps[3], kernel_size=3, stride=1, padding=1))
    sq.add_module('maxpool2', builder.Maxpool2d(kernel_size=2))
    sq.add_module('conv5',
                  builder.Conv2dBNReLU(in_channels=deps[3], out_channels=deps[4], kernel_size=3, stride=1, padding=1))
    sq.add_module('conv6',
                  builder.Conv2dBNReLU(in_channels=deps[4], out_channels=deps[5], kernel_size=3, stride=1, padding=1))
    sq.add_module('conv7',
                  builder.Conv2dBNReLU(in_channels=deps[5], out_channels=deps[6], kernel_size=3, stride=1, padding=1))
    sq.add_module('maxpool3', builder.Maxpool2d(kernel_size=2))
    sq.add_module('conv8',
                  builder.Conv2dBNReLU(in_channels=deps[6], out_channels=deps[7], kernel_size=3, stride=1, padding=1))
    sq.add_module('conv9',
                  builder.Conv2dBNReLU(in_channels=deps[7], out_channels=deps[8], kernel_size=3, stride=1, padding=1))
    sq.add_module('conv10',
                  builder.Conv2dBNReLU(in_channels=deps[8], out_channels=deps[9], kernel_size=3, stride=1, padding=1))
    sq.add_module('maxpool4', builder.Maxpool2d(kernel_size=2))
    sq.add_module('conv11',
                  builder.Conv2dBNReLU(in_channels=deps[9], out_channels=deps[10], kernel_size=3, stride=1, padding=1))
    sq.add_module('conv12',
                  builder.Conv2dBNReLU(in_channels=deps[10], out_channels=deps[11], kernel_size=3, stride=1, padding=1))
    sq.add_module('conv13',
                  builder.Conv2dBNReLU(in_channels=deps[11], out_channels=deps[12], kernel_size=3, stride=1, padding=1))
    sq.add_module('maxpool5', builder.Maxpool2d(kernel_size=2))
    return sq

class VCNet(nn.Module):

    def __init__(self, num_classes, builder:ConvBuilder, deps):
        super(VCNet, self).__init__()
        if deps is None:
            deps = VGG_ORIGIN_DEPS
        self.stem = _create_vgg_stem(builder=builder, deps=deps)
        self.flatten = builder.Flatten()
        self.linear1 = builder.IntermediateLinear(in_features=deps[12], out_features=512)
        self.relu = builder.ReLU()
        self.linear2 = builder.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        out = self.stem(x)
        out = self.flatten(out)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        return out


def create_vc(cfg, builder):
    return VCNet(num_classes=10, builder=builder, deps=cfg.deps)
def create_vh(cfg, builder):
    return VCNet(num_classes=100, builder=builder, deps=cfg.deps)
