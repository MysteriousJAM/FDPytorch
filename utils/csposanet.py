import torch
import torch.nn.functional as F
import torch.nn as nn
import math

from utils.layers import ConvBnActivation


class OSABlock(nn.Module):
    def __init__(self, in_channels, activation, stride=2, keep_in_channels=False):
        super(OSABlock, self).__init__()
        assert in_channels % 2 == 0
        self.channels = in_channels // 2
        assert isinstance(stride, int)
        self.stride = stride

        self.conv0 = ConvBnActivation(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride,
                                      activation=activation)
        if self.stride != 1:
            self.conv1 = ConvBnActivation(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1,
                                          activation=activation)
        self.conv2 = ConvBnActivation(in_channels // 2, in_channels // 2, kernel_size=3, stride=1,
                                      activation=activation)
        self.conv3 = ConvBnActivation(in_channels // 2, in_channels // 2, kernel_size=3, stride=1,
                                      activation=activation)
        self.conv4 = ConvBnActivation(in_channels, in_channels, kernel_size=1, stride=1, activation=activation)
        if keep_in_channels:
            self.conv5 = ConvBnActivation(2 * in_channels, in_channels, kernel_size=1, stride=1,
                                          activation=activation)
        else:
            self.conv5 = ConvBnActivation(2 * in_channels, 2 * in_channels, kernel_size=1, stride=1,
                                          activation=activation)

    def forward(self, x):
        x = self.conv0(x)

        if self.stride != 1:
            x = self.conv1(x)
        skip1 = x

        x = torch.split(x, self.channels, dim=1)[1]
        x = self.conv2(x)
        skip2 = x

        x = self.conv3(x)
        x = torch.cat([skip2, x], dim=1)

        x = self.conv4(x)
        x = torch.cat([skip1, x], dim=1)

        x = self.conv5(x)
        return x


class CSPOSANet(nn.Module):
    def __init__(self, activation='leaky', num_features=2, multiplier=1.0):
        super(CSPOSANet, self).__init__()
        assert num_features <= 4
        self.num_features = num_features
        self.conv1 = ConvBnActivation(3, int(32 * multiplier), kernel_size=3, stride=2, activation=activation)
        self.conv2 = ConvBnActivation(int(32 * multiplier), int(64 * multiplier), kernel_size=3, stride=1,
                                      activation=activation)

        self.block1 = OSABlock(int(64 * multiplier), activation)
        self.block2 = OSABlock(int(128 * multiplier), activation)
        self.block3 = nn.Sequential(OSABlock(int(256 * multiplier), activation, keep_in_channels=True),
                                    OSABlock(int(256 * multiplier), activation, stride=1, keep_in_channels=True),
                                    OSABlock(int(256 * multiplier), activation, stride=1))
        self.block4 = OSABlock(int(512 * multiplier), activation)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)

        feats = [None] * 4
        feats[0] = self.block1(x)
        feats[1] = self.block2(feats[0])
        feats[2] = self.block3(feats[1])
        feats[3] = self.block4(feats[2])
        return feats[-self.num_features:]


class CSPOSANetClassification(nn.Module):
    def __init__(self, activation='leaky', multiplier=1.0, classes=1000):
        super(CSPOSANetClassification, self).__init__()
        self.conv1 = ConvBnActivation(3, int(32 * multiplier), kernel_size=3, stride=2, activation=activation)
        self.conv2 = ConvBnActivation(int(32 * multiplier), int(64 * multiplier), kernel_size=3, stride=2,
                                      activation=activation)

        self.block1 = OSABlock(int(64 * multiplier), activation)
        self.block2 = OSABlock(int(128 * multiplier), activation)
        self.block3 = OSABlock(int(256 * multiplier), activation)

        self.conv3 = ConvBnActivation(int(512 * multiplier), int(512 * multiplier), kernel_size=3, stride=1,
                                      activation=activation)

        self.fc = nn.Linear(int(512 * multiplier), classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x, _ = self.block1(x)
        x, _ = self.block2(x)
        x, _ = self.block3(x)
        x = self.conv3(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = self.fc(x.flatten(start_dim=1))
        return x
