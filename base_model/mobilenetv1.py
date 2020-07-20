import torch.nn as nn
from builder import ConvBuilder
from constants import MI1_ORIGIN_DEPS

class MobileV1Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, builder:ConvBuilder, in_planes, out_planes, stride=1):
        super(MobileV1Block, self).__init__()
        self.depthwise = builder.Conv2dBNReLU(in_channels=in_planes, out_channels=in_planes, kernel_size=3,
                                          stride=stride, padding=1, groups=in_planes)
        self.pointwise = builder.Conv2dBNReLU(in_channels=in_planes, out_channels=out_planes, kernel_size=1,
                                          stride=1, padding=0)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


cifar_cfg = [16, 32, 32, 64, 64, (128,2), 128, 128, 128, 128, 128, (256,2), 256]    # 93



class MobileV1CifarNet(nn.Module):

    def __init__(self, builder:ConvBuilder, num_classes):
        super(MobileV1CifarNet, self).__init__()
        self.conv1 = builder.Conv2dBNReLU(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        blocks = []
        in_planes = cifar_cfg[0]
        for x in cifar_cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            blocks.append(MobileV1Block(builder=builder, in_planes=in_planes, out_planes=out_planes, stride=stride))
            in_planes = out_planes
        self.stem = builder.Sequential(*blocks)
        self.gap = builder.GAP(kernel_size=8)
        self.linear = builder.Linear(cifar_cfg[-1], num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.stem(out)
        out = self.gap(out)
        out = self.linear(out)
        return out

class MobileV1ImagenetNet(nn.Module):

    def __init__(self, builder:ConvBuilder, num_classes, deps=None):
        super(MobileV1ImagenetNet, self).__init__()
        if deps is None:
            deps = MI1_ORIGIN_DEPS
        assert len(deps) == 27
        self.conv1 = builder.Conv2dBNReLU(in_channels=3, out_channels=deps[0], kernel_size=3, stride=2, padding=1)
        blocks = []
        for block_idx in range(13):
            depthwise_channels = int(deps[block_idx * 2 + 1])
            pointwise_channels = int(deps[block_idx * 2 + 2])
            stride = 2 if block_idx in [1, 3, 5, 11] else 1
            blocks.append(MobileV1Block(builder=builder, in_planes=depthwise_channels, out_planes=pointwise_channels, stride=stride))

        self.stem = builder.Sequential(*blocks)
        self.gap = builder.GAP(kernel_size=7)
        self.linear = builder.Linear(deps[-1], num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.stem(out)
        out = self.gap(out)
        out = self.linear(out)
        return out

def create_MobileV1Cifar(cfg, builder):
    return MobileV1CifarNet(builder=builder, num_classes=10)
def create_MobileV1CH(cfg, builder):
    return MobileV1CifarNet(builder=builder, num_classes=100)
def create_MobileV1Imagenet(cfg, builder):
    return MobileV1ImagenetNet(builder=builder, num_classes=1000, deps=cfg.deps)