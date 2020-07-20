import torch.nn as nn
from builder import ConvBuilder

class NoBNBuilder(ConvBuilder):

    def __init__(self, base_config):
        super(NoBNBuilder, self).__init__(base_config=base_config)
        print('NoBN ConvBuilder initialized.')

    def Conv2dBN(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', use_original_conv=False):
        conv_layer = self.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, dilation=dilation, groups=groups,
                                 bias=True, padding_mode=padding_mode, use_original_conv=use_original_conv)
        se = self.Sequential()
        se.add_module('conv', conv_layer)
        return se






