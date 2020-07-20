'''
ResNet in PyTorch.absFor Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Deep Residual Learning for Image Recognition. arXiv:1512.03385

Note: cifar_resnet18 constructs the same model with that from
https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''

import torch.nn as nn
from builder import ConvBuilder

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, builder:ConvBuilder, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.bd = builder
        self.relu = builder.ReLU()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = builder.Conv2dBN(in_channels=in_planes, out_channels=self.expansion * planes, kernel_size=1, stride=stride)
        else:
            self.shortcut = builder.ResIdentity(num_channels=in_planes)

        self.conv1 = builder.Conv2dBNReLU(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1)
        self.conv2 = builder.Conv2dBN(in_channels=planes, out_channels=self.expansion * planes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bd.add(out, self.shortcut(x))
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, builder:ConvBuilder, block, num_blocks, num_classes=10, width_multiplier=None):
        super(ResNet, self).__init__()

        print('width multiplier: ', width_multiplier)

        if width_multiplier is None:
            width_multiplier = 1
        else:
            width_multiplier = width_multiplier[0]

        self.bd = builder
        self.in_planes = int(64 * width_multiplier)
        self.conv1 = builder.Conv2dBNReLU(3, int(64 * width_multiplier), kernel_size=7, stride=2, padding=3)
        self.stage1 = self._make_stage(block, int(64 * width_multiplier), num_blocks[0], stride=1)
        self.stage2 = self._make_stage(block, int(128 * width_multiplier), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(block, int(256 * width_multiplier), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(block, int(512 * width_multiplier), num_blocks[3], stride=2)
        self.gap = builder.GAP(kernel_size=7)
        self.linear = self.bd.Linear(int(512*block.expansion*width_multiplier), num_classes)

    def _make_stage(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            blocks.append(block(builder=self.bd, in_planes=self.in_planes, planes=int(planes), stride=stride))
            self.in_planes = int(planes * block.expansion)
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bd.max_pool2d(out, kernel_size=3, stride=2, padding=1)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = self.linear(out)
        return out

def create_ResNet18(cfg, builder):
    return ResNet(builder, BasicBlock, [2,2,2,2], num_classes=1000, width_multiplier=cfg.deps)
def create_ResNet34(cfg, builder):
    return ResNet(builder, BasicBlock, [3,4,6,3], num_classes=1000, width_multiplier=cfg.deps)