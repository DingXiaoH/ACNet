import torch.nn as nn
from builder import ConvBuilder
from constants import wrn_convert_flattened_deps

class WRNCifarBlock(nn.Module):

    def __init__(self, input_channels, block_channels, stride, projection_shortcut, use_dropout, builder:ConvBuilder):
        super(WRNCifarBlock, self).__init__()
        assert len(block_channels) == 2

        if projection_shortcut:
            self.proj = builder.BNReLUConv2d(in_channels=input_channels, out_channels=block_channels[1], kernel_size=1, stride=stride, padding=0)
        else:
            self.proj = builder.ResIdentity(num_channels=block_channels[1])

        self.conv1 = builder.BNReLUConv2d(in_channels=input_channels, out_channels=block_channels[0], kernel_size=3,
                                          stride=stride, padding=1)
        if use_dropout:
            self.dropout = builder.Dropout(keep_prob=0.7)
            print('use dropout for WRN')
        else:
            self.dropout = builder.Identity()
        self.conv2 = builder.BNReLUConv2d(in_channels=block_channels[0], out_channels=block_channels[1], kernel_size=3,
                                          stride=1, padding=1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.dropout(x)
        x = self.conv2(x)
        x += self.proj(input)
        return x

class WRNCifarNet(nn.Module):

    def __init__(self, block_counts, num_classes, builder:ConvBuilder, deps, use_dropout):
        super(WRNCifarNet, self).__init__()
        self.bd = builder
        converted_deps = wrn_convert_flattened_deps(deps)
        print('the converted deps is ', converted_deps)

        self.conv1 = builder.Conv2d(in_channels=3, out_channels=converted_deps[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.stage1 = self._build_wrn_stage(num_blocks=block_counts[0], stage_input_channels=converted_deps[0],
                                            stage_deps=converted_deps[1], downsample=False, use_dropout=use_dropout)
        self.stage2 = self._build_wrn_stage(num_blocks=block_counts[1], stage_input_channels=converted_deps[1][-1][1],
                                            stage_deps=converted_deps[2], downsample=True, use_dropout=use_dropout)
        self.stage3 = self._build_wrn_stage(num_blocks=block_counts[2], stage_input_channels=converted_deps[2][-1][1],
                                            stage_deps=converted_deps[3], downsample=True, use_dropout=use_dropout)
        self.last_bn = builder.BatchNorm2d(num_features=converted_deps[3][-1][1])
        self.linear = builder.Linear(in_features=converted_deps[3][-1][1], out_features=num_classes)


    def _build_wrn_stage(self, num_blocks, stage_input_channels, stage_deps, downsample, use_dropout):
        se = self.bd.Sequential()
        for i in range(num_blocks):
            if i == 0:
                block_input_channels = stage_input_channels
            else:
                block_input_channels = stage_deps[i - 1][1]
            if i == 0 and downsample:
                stride = 2
            else:
                stride = 1
            se.add_module(name='block{}'.format(i+1),
                          module=WRNCifarBlock(input_channels=block_input_channels, block_channels=stage_deps[i],
                                               stride=stride, projection_shortcut=(i==0), use_dropout=use_dropout, builder=self.bd))
        return se

    def forward(self, x):
        out = self.conv1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.last_bn(out)
        out = self.bd.avg_pool2d(in_features=out, kernel_size=8, stride=1, padding=0)
        out = self.bd.flatten(out)
        out = self.linear(out)
        return out



def create_wrnc16plain(cfg, builder):
    return WRNCifarNet(block_counts=(2,2,2), num_classes=10, builder=builder, deps=cfg.deps, use_dropout=False)
def create_wrnc16drop(cfg, builder):
    return WRNCifarNet(block_counts=(2,2,2), num_classes=10, builder=builder, deps=cfg.deps, use_dropout=True)
def create_wrnc28plain(cfg, builder):
    return WRNCifarNet(block_counts=(4,4,4), num_classes=10, builder=builder, deps=cfg.deps, use_dropout=False)
def create_wrnc28drop(cfg, builder):
    return WRNCifarNet(block_counts=(4,4,4), num_classes=10, builder=builder, deps=cfg.deps, use_dropout=True)
def create_wrnc40plain(cfg, builder):
    return WRNCifarNet(block_counts=(6,6,6), num_classes=10, builder=builder, deps=cfg.deps, use_dropout=False)
def create_wrnc40drop(cfg, builder):
    return WRNCifarNet(block_counts=(6,6,6), num_classes=10, builder=builder, deps=cfg.deps, use_dropout=True)

def create_wrnh16plain(cfg, builder):
    return WRNCifarNet(block_counts=(2,2,2), num_classes=100, builder=builder, deps=cfg.deps, use_dropout=False)
def create_wrnh16drop(cfg, builder):
    return WRNCifarNet(block_counts=(2,2,2), num_classes=100, builder=builder, deps=cfg.deps, use_dropout=True)
def create_wrnh28plain(cfg, builder):
    return WRNCifarNet(block_counts=(4,4,4), num_classes=100, builder=builder, deps=cfg.deps, use_dropout=False)
def create_wrnh28drop(cfg, builder):
    return WRNCifarNet(block_counts=(4,4,4), num_classes=100, builder=builder, deps=cfg.deps, use_dropout=True)
def create_wrnh40plain(cfg, builder):
    return WRNCifarNet(block_counts=(6,6,6), num_classes=100, builder=builder, deps=cfg.deps, use_dropout=False)
def create_wrnh40drop(cfg, builder):
    return WRNCifarNet(block_counts=(6,6,6), num_classes=100, builder=builder, deps=cfg.deps, use_dropout=True)