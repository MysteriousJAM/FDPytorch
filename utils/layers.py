import torch

import torch.nn as nn
import torch.nn.functional as F


class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class ConvBnActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation=1, groups=1, bias=False, bn=True,
                 activation='relu'):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.module_list = nn.ModuleList()
        self.module_list.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=pad, dilation=dilation, groups=groups, bias=bias))
        if bn:
            self.module_list.append(nn.BatchNorm2d(out_channels))
        if activation == "mish":
            self.module_list.append(Mish())
        elif activation == "relu":
            self.module_list.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.module_list.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            raise NotImplementedError

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x

