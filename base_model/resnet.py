'''
ResNet in PyTorch.absFor Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Deep Residual Learning for Image Recognition. arXiv:1512.03385

Note: cifar_resnet18 constructs the same model with that from
https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''

import torch.nn as nn
import torch.nn.functional as F
from builder import ConvBuilder

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, builder=None):
        super(Bottleneck, self).__init__()
        self.bd = builder

        self.conv1 = self.bd.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = self.bd.BatchNorm2d(planes)
        self.conv2 = self.bd.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = self.bd.BatchNorm2d(planes)
        self.conv3 = self.bd.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = self.bd.BatchNorm2d(self.expansion*planes)

        self.shortcut = self.bd.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = self.bd.Sequential(
                self.bd.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                self.bd.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class RCBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, builder=None):
        super(RCBlock, self).__init__()
        self.bd = builder

        self.conv1 = self.bd.Conv2dBNReLU(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = self.bd.Conv2dBN(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.shortcut = self.bd.Sequential()
        if stride != 1:
            self.shortcut = self.bd.Conv2dBN(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.bd.relu(out)
        return out


class RCNet(nn.Module):

    def __init__(self, block_counts, num_classes, builder:ConvBuilder):
        super(RCNet, self).__init__()
        self.bd = builder

        self.conv1 = self.bd.Conv2dBNReLU(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.stage1 = self._build_stage(stage_in_channels=16, out_channels=16, num_blocks=block_counts[0], stride=1)
        self.stage2 = self._build_stage(stage_in_channels=16, out_channels=32, num_blocks=block_counts[1], stride=2)
        self.stage3 = self._build_stage(stage_in_channels=32, out_channels=64, num_blocks=block_counts[2], stride=2)
        self.linear = self.bd.Linear(in_features=64, out_features=num_classes)


    def _build_stage(self, stage_in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        in_channel_list = [stage_in_channels] + [out_channels] * (num_blocks - 1)
        layers = []
        for block_stride, block_in_channels in zip(strides, in_channel_list):
            layers.append(RCBlock(in_channels=block_in_channels, out_channels=out_channels, stride=block_stride, builder=self.bd))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.bd.avg_pool2d(in_features=out, kernel_size=8, stride=1, padding=0)
        out = self.bd.flatten(out)
        out = self.linear(out)
        return out


def create_RC56(cfg, builder):
    return RCNet(block_counts=[9,9,9], num_classes=10, builder=builder)

def create_RC110(cfg, builder):
    return RCNet(block_counts=[18,18,18], num_classes=10, builder=builder)

def create_RC164(cfg, builder):
    return RCNet(block_counts=[27,27,27], num_classes=10, builder=builder)




class ResNet(nn.Module):
    def __init__(self, block, num_blocks, builder, num_classes=10):
        super(ResNet, self).__init__()
        self.bd = builder

        self.in_planes = 64

        self.conv1 = self.bd.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = self.bd.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = self.bd.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, builder=self.bd))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bd.relu(self.bn1(self.conv1(x)))
        out = self.bd.max_pool2d(out, kernel_size=3, stride=2, padding=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.bd.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def create_ResNet34(cfg, builder):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=1000, builder=builder)

def create_ResNet50(cfg, builder):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=1000, builder=builder)

def create_ResNet101(cfg, builder):
    return ResNet(Bottleneck, [3,4,23,3], num_classes=1000, builder=builder)

def create_ResNet152(cfg, builder):
    return ResNet(Bottleneck, [3,8,36,3], num_classes=1000, builder=builder)

