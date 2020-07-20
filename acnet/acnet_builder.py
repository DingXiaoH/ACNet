from builder import ConvBuilder
from acnet.acb import ACBlock
import torch.nn as nn

class ACNetBuilder(ConvBuilder):

    def __init__(self, base_config, deploy, gamma_init=None):
        super(ACNetBuilder, self).__init__(base_config=base_config)
        self.deploy = deploy
        self.use_last_bn = False
        self.gamma_init = gamma_init

    def switch_to_deploy(self):
        self.deploy = True

    def Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', use_original_conv=False):
        if use_original_conv or kernel_size == 1 or kernel_size == (1, 1) or kernel_size >= 7:
            return super(ACNetBuilder, self).Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                 padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, use_original_conv=True)
        else:
            return ACBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                       padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, deploy=self.deploy,
                           use_last_bn=self.use_last_bn, gamma_init=self.gamma_init)


    def Conv2dBN(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', use_original_conv=False):
        if use_original_conv or kernel_size == 1 or kernel_size == (1, 1) or kernel_size >= 7:
            return super(ACNetBuilder, self).Conv2dBN(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                 padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, use_original_conv=True)
        else:
            return ACBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                       padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, deploy=self.deploy,
                           use_last_bn=self.use_last_bn, gamma_init=self.gamma_init)


    def Conv2dBNReLU(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', use_original_conv=False):
        if use_original_conv or kernel_size == 1 or kernel_size == (1, 1) or kernel_size >= 7:
            return super(ACNetBuilder, self).Conv2dBNReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                 padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, use_original_conv=True)
        else:
            se = nn.Sequential()
            se.add_module('acb', ACBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                       padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, deploy=self.deploy,
                                         use_last_bn=self.use_last_bn, gamma_init=self.gamma_init))
            se.add_module('relu', self.ReLU())
            return se


    def BNReLUConv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', use_original_conv=False):
        if use_original_conv or kernel_size == 1 or kernel_size == (1, 1) or kernel_size >= 7:
            return super(ACNetBuilder, self).BNReLUConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                 padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, use_original_conv=True)
        bn_layer = self.BatchNorm2d(num_features=in_channels)
        conv_layer = ACBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                       padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, deploy=self.deploy)
        se = self.Sequential()
        se.add_module('bn', bn_layer)
        se.add_module('relu', self.ReLU())
        se.add_module('acb', conv_layer)
        return se