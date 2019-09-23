from builder import ConvBuilder
import torch.nn as nn


class ACBlock(nn.Module):

    #   TODO The padding configurations work on ResNet-56 and WRN (3x3, pad 1).
    #   May not work on other models
    #   In general cases, I know how to do the math for the asymmetric layers to #match# the target feature map
    #   size produced by the square conv layer, which requires me to know the size of outputs of square conv layer #in advance#.
    #   Maybe I will come up with a general solution in the future (when I get familiar with PyTorch).
    #   Of course, case-by-case solutions are obvious.
    #   by Ding Xiaohan 2019/08/30
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros'):
        super(ACBlock, self).__init__()
        self.square_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size,kernel_size), stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode)
        self.square_BN = nn.BatchNorm2d(num_features=out_channels)

        self.vertical_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
                                      stride=stride,
                                      padding=(min(1, padding), 0), dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)
        self.vertical_BN = nn.BatchNorm2d(num_features=out_channels)
        self.horizontal_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                      stride=stride,
                                      padding=(0, min(1, padding)), dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)
        self.horizontal_BN = nn.BatchNorm2d(num_features=out_channels)


    def forward(self, input):
        square_outputs = self.square_layer(input)
        square_outputs = self.square_BN(square_outputs)
        # return square_outputs
        vertical_outputs = self.vertical_layer(input)
        vertical_outputs = self.vertical_BN(vertical_outputs)
        # print(vertical_outputs.size())
        horizontal_outputs = self.horizontal_layer(input)
        horizontal_outputs = self.horizontal_BN(horizontal_outputs)
        # print(horizontal_outputs.size())
        return square_outputs + vertical_outputs + horizontal_outputs






class ACNetBuilder(ConvBuilder):

    def __init__(self):
        super(ACNetBuilder, self).__init__()

    def Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', use_original_conv=False):
        if use_original_conv or kernel_size == 1 or kernel_size == (1, 1):
            return super(ACNetBuilder, self).Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                 padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, use_original_conv=True)
        else:
            return ACBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                       padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode)


    def Conv2dBN(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', use_original_conv=False):
        if use_original_conv or kernel_size == 1 or kernel_size == (1, 1):
            return super(ACNetBuilder, self).Conv2dBN(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                 padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, use_original_conv=True)
        else:
            return ACBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                       padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode)

    def BNReLUConv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', use_original_conv=False):
        if use_original_conv or kernel_size == 1 or kernel_size == (1, 1):
            return super(ACNetBuilder, self).BNReLUConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                 padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, use_original_conv=True)
        bn_layer = self.BatchNorm2d(num_features=in_channels)
        conv_layer = ACBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                       padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode)
        se = self.Sequential()
        se.add_module('bn', bn_layer)
        se.add_module('relu', self.ReLU())
        se.add_module('conv', conv_layer)
        return se