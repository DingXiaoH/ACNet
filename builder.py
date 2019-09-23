import torch.nn as nn
import torch.nn.functional as F

class ConvBuilder(nn.Module):

    def __init__(self):
        super(ConvBuilder, self).__init__()
        print('ConvBuilder initialized.')
        self.BN_eps = 1e-5
        self.BN_momentum = 0.1
        self.BN_affine = True
        self.BN_track_running_stats = True

    def set_BN_config(self, eps, momentum, affine, track_running_stats):
        self.BN_eps = eps
        self.BN_momentum = momentum
        self.BN_afine = affine
        self.BN_track_running_stats = track_running_stats


    def Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', use_original_conv=False):
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)

    # The running estimates are kept with a default momentum of 0.1.
    # By default, the elements of \gammaγ are sampled from \mathcal{U}(0, 1)U(0,1) and the elements of \betaβ are set to 0.
    # If track_running_stats is set to False, this layer then does not keep running estimates, and batch statistics are instead used during evaluation time as well.
    def BatchNorm2d(self, num_features, eps=None, momentum=None, affine=None, track_running_stats=None):
        if eps is None:
            eps = self.BN_eps
        if momentum is None:
            momentum = self.BN_momentum
        if affine is None:
            affine = self.BN_affine
        if track_running_stats is None:
            track_running_stats = self.BN_track_running_stats
        return nn.BatchNorm2d(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)


    def Sequential(self, *args):
        return nn.Sequential(*args)

    def ReLU(self):
        return nn.ReLU()


    def Conv2dBN(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', use_original_conv=False):
        conv_layer = self.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode, use_original_conv=use_original_conv)
        bn_layer = self.BatchNorm2d(num_features=out_channels)
        se = self.Sequential()
        se.add_module('conv', conv_layer)
        se.add_module('bn', bn_layer)
        return se

    def Conv2dBNReLU(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', use_original_conv=False):
        conv = self.Conv2dBN(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                 padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, use_original_conv=use_original_conv)
        conv.add_module('relu', self.ReLU())
        return conv

    def BNReLUConv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', use_original_conv=False):
        bn_layer = self.BatchNorm2d(num_features=in_channels)
        conv_layer = self.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode)
        se = self.Sequential()
        se.add_module('bn', bn_layer)
        se.add_module('relu', self.ReLU())
        se.add_module('conv', conv_layer)
        return se

    def Linear(self, in_features, out_features, bias=True):
        return nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

    def Identity(self):
        return nn.Identity()

    def Dropout(self, keep_prob):
        return nn.Dropout(p=1-keep_prob)


    def relu(self, in_features):
        return F.relu(in_features)

    def max_pool2d(self, in_features, kernel_size, stride, padding):
        return F.max_pool2d(in_features, kernel_size=kernel_size, stride=stride, padding=padding)

    def avg_pool2d(self, in_features, kernel_size, stride, padding):
        return F.avg_pool2d(in_features, kernel_size=kernel_size, stride=stride, padding=padding)

    def flatten(self, in_features):
        result = in_features.view(in_features.size(0), -1)
        return result









