import torch.nn as nn
from builder import ConvBuilder

class MobileV1Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, builder:ConvBuilder, in_planes, out_planes, stride=1):
        super(MobileV1Block, self).__init__()
        self.conv1 = builder.Conv2dBNReLU(in_channels=in_planes, out_channels=in_planes, kernel_size=3,
                                          stride=stride, padding=1, groups=in_planes)
        self.conv2 = builder.Conv2dBNReLU(in_channels=in_planes, out_channels=out_planes, kernel_size=1,
                                          stride=1, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out

imagenet_cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
# cifar_cfg = [16, (32,2), 32, (64,2), 64, (128,2), 128, 128, 128, 128, 128, (256,2), 256]      # 86%
# cifar_cfg = [16, 32, 32, (64,2), 64, (128,2), 128, 128, 128, 128, 128, (256,2), 256]
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
        self.linear = builder.Linear(cifar_cfg[-1], num_classes)
        self.bd = builder



    def forward(self, x):
        out = self.conv1(x)
        out = self.stem(out)
        out = self.bd.avg_pool2d(out, 8, stride=1, padding=0)
        out = self.bd.flatten(out)
        out = self.linear(out)
        return out

def create_MobileV1Cifar(cfg, builder):
    return MobileV1CifarNet(builder=builder, num_classes=10)
def create_MobileV1CH(cfg, builder):
    return MobileV1CifarNet(builder=builder, num_classes=100)