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
from constants import RESNET50_ORIGIN_DEPS_FLATTENED, resnet_bottleneck_origin_deps_flattened, rc_convert_flattened_deps, rc_origin_deps_flattened
import torch
import numpy as np
from utils.misc import efficient_torchvision_style_normalize
import torch.nn.functional as F

class BottleneckBranch(nn.Module):

    def __init__(self, builder:ConvBuilder, in_channels, deps, stride=1):
        super(BottleneckBranch, self).__init__()
        assert len(deps) == 3
        self.conv1 = builder.Conv2dBNReLU(in_channels, deps[0], kernel_size=1)
        self.conv2 = builder.Conv2dBNReLU(deps[0], deps[1], kernel_size=3, stride=stride, padding=1)
        self.conv3 = builder.Conv2dBN(deps[1], deps[2], kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

class BasicBranch(nn.Module):

    def __init__(self, builder:ConvBuilder, in_channels, deps, stride=1):
        super(BasicBranch, self).__init__()
        assert len(deps) == 2
        self.conv1 = builder.Conv2dBNReLU(in_channels, deps[0], kernel_size=3, stride=stride, padding=1)
        self.conv2 = builder.Conv2dBN(deps[0], deps[1], kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class ResNetBottleneckStage(nn.Module):

    #   stage_deps:     3n+1 (first is the projection),  n is the num of blocks

    def __init__(self, builder:ConvBuilder, in_planes, stage_deps, stride=1):
        super(ResNetBottleneckStage, self).__init__()
        print('building stage: in {}, deps {}'.format(in_planes, stage_deps))
        assert (len(stage_deps) - 1) % 3 == 0
        self.num_blocks = (len(stage_deps) - 1) // 3
        stage_out_channels = stage_deps[3]
        for i in range(2, self.num_blocks):
            assert stage_deps[3 * i] == stage_out_channels

        self.relu = builder.ReLU()

        self.projection = builder.Conv2dBN(in_channels=in_planes, out_channels=stage_deps[0], kernel_size=1, stride=stride)
        self.align_opr = builder.ResNetAlignOpr(channels=stage_deps[0])

        for i in range(self.num_blocks):
            in_c = in_planes if i == 0 else stage_out_channels
            block_stride = stride if i == 0 else 1
            self.__setattr__('block{}'.format(i), BottleneckBranch(builder=builder,
                            in_channels=in_c, deps=stage_deps[1+i*3: 4+i*3], stride=block_stride))

    def forward(self, x):
        proj = self.align_opr(self.projection(x))
        out = proj + self.align_opr(self.block0(x))
        out = self.relu(out)
        for i in range(1, self.num_blocks):
            out = out + self.align_opr(self.__getattr__('block{}'.format(i))(out))
            out = self.relu(out)
        return out



class ResNetBasicStage(nn.Module):

    def __init__(self, builder:ConvBuilder, in_planes, stage_deps, stride=1, is_first=False):
        super(ResNetBasicStage, self).__init__()
        print('building stage: in {}, deps {}'.format(in_planes, stage_deps))
        self.num_blocks = len(stage_deps) // 2

        stage_out_channels = stage_deps[0]
        for i in range(0, self.num_blocks):
            assert stage_deps[i * 2 + 2] == stage_out_channels

        if is_first:
            self.conv1 = builder.Conv2dBN(in_channels=in_planes, out_channels=stage_out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.projection = builder.Conv2dBN(in_channels=in_planes, out_channels=stage_out_channels,
                                               kernel_size=1, stride=stride)

        self.relu = builder.ReLU()
        self.align_opr = builder.ResNetAlignOpr(channels=stage_out_channels)

        for i in range(self.num_blocks):
            if i == 0 and is_first:
                in_c = stage_deps[0]
            elif i == 0:
                in_c = in_planes
            else:
                in_c = stage_out_channels
            block_stride = stride if i == 0 else 1
            self.__setattr__('block{}'.format(i), BasicBranch(builder=builder,
                            in_channels=in_c, deps=stage_deps[1 + i*2: 3 + i*2],
                            stride=block_stride))

    def forward(self, x):
        if hasattr(self, 'conv1'):
            base_out = self.relu(self.align_opr(self.conv1(x)))
            out = base_out + self.align_opr(self.block0(base_out))
        else:
            proj = self.align_opr(self.projection(x))
            out = proj + self.align_opr(self.block0(x))

        out = self.relu(out)
        for i in range(1, self.num_blocks):
            out = out + self.align_opr(self.__getattr__('block{}'.format(i))(out))
            out = self.relu(out)
        return out


class SBottleneckResNet(nn.Module):
    def __init__(self, builder:ConvBuilder, num_blocks, num_classes=1000, deps=None):
        super(SBottleneckResNet, self).__init__()

        if deps is None:
            if num_blocks == [3,4,6,3]:
                deps = RESNET50_ORIGIN_DEPS_FLATTENED
            elif num_blocks == [3,4,23,3]:
                deps = resnet_bottleneck_origin_deps_flattened(101)
            else:
                raise ValueError('???')

        self.conv1 = builder.Conv2dBNReLU(3, deps[0], kernel_size=7, stride=2, padding=3)
        self.maxpool = builder.Maxpool2d(kernel_size=3, stride=2, padding=1)
        #   every stage has  num_block * 3 + 1
        nls = [n*3+1 for n in num_blocks]    # num layers in each stage
        self.stage1 = ResNetBottleneckStage(builder=builder, in_planes=deps[0], stage_deps=deps[1: nls[0]+1])
        self.stage2 = ResNetBottleneckStage(builder=builder, in_planes=deps[nls[0]],
                                            stage_deps=deps[nls[0]+1: nls[0]+1+nls[1]], stride=2)
        self.stage3 = ResNetBottleneckStage(builder=builder, in_planes=deps[nls[0]+nls[1]],
                                            stage_deps=deps[nls[0]+nls[1]+1: nls[0]+1+nls[1]+nls[2]], stride=2)
        self.stage4 = ResNetBottleneckStage(builder=builder, in_planes=deps[nls[0] + nls[1] + nls[2]],
                                            stage_deps=deps[nls[0] + nls[1] + nls[2] + 1: nls[0] + 1 + nls[1] + nls[2] + nls[3]],
                                            stride=2)
        self.gap = builder.GAP(kernel_size=7)
        self.fc = builder.Linear(deps[-1], num_classes)
        # self.mean = 0
        # self.batches = 0


    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = self.fc(out)
        return out


class SRCNet(nn.Module):

    def __init__(self, block_counts, num_classes, builder:ConvBuilder, deps):
        super(SRCNet, self).__init__()
        self.bd = builder
        assert block_counts[0] == block_counts[1]
        assert block_counts[1] == block_counts[2]
        if deps is None:
            deps = rc_origin_deps_flattened(block_counts[0])

        assert len(deps) == block_counts[0] * 6 + 3
        filters_per_stage = len(deps) // 3

        self.stage1 = ResNetBasicStage(builder=builder, in_planes=3,
                                       stage_deps=deps[0:filters_per_stage], stride=1, is_first=True)
        self.stage2 = ResNetBasicStage(builder=builder, in_planes=deps[filters_per_stage - 1],
                                       stage_deps=deps[filters_per_stage : 2 * filters_per_stage], stride=2)
        self.stage3 = ResNetBasicStage(builder=builder, in_planes=deps[2 * filters_per_stage - 1],
                                       stage_deps=deps[2 * filters_per_stage :], stride=2)
        self.gap = self.bd.GAP(kernel_size=8)
        self.linear = self.bd.Linear(in_features=deps[-1], out_features=num_classes)

    def forward(self, x):
        out = self.stage1(x)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.gap(out)
        out = self.linear(out)
        return out


def create_SResNet50(cfg, builder):
    return SBottleneckResNet(builder, [3,4,6,3], num_classes=1000, deps=cfg.deps)

def create_SResNet101(cfg, builder):
    return SBottleneckResNet(builder, [3,4,23,3], num_classes=1000, deps=cfg.deps)

def create_SResNet152(cfg, builder):
    return SBottleneckResNet(builder, [3,8,36,3], num_classes=1000, deps=cfg.deps)

def create_SRC56(cfg, builder):
    return SRCNet(block_counts=[9, 9, 9], num_classes=10, builder=builder, deps=cfg.deps)

def create_SRC110(cfg, builder):
    return SRCNet(block_counts=[18, 18, 18], num_classes=10, builder=builder, deps=cfg.deps)

def create_SRC164(cfg, builder):
    return SRCNet(block_counts=[27, 27, 27], num_classes=10, builder=builder, deps=cfg.deps)