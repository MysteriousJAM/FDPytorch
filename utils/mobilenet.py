from torch import nn


def _make_divisible(value, divisor, min_value=None):
    """

    :param value:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int((value + divisor / 2) // divisor * divisor))
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


class Activation(nn.Module):
    def __init__(self, act_type='relu'):
        """

        :param act_type:
        """
        super(Activation, self).__init__()
        self.act_type = act_type.lower()
        if self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'relu6':
            self.act = nn.ReLU6(inplace=True)
        elif self.act_type == 'hardswish':
            self.act = nn.Hardswish()
        elif self.act_type == 'mish':
            self.act = nn.Sequential(nn.Tanh(), nn.Softplus())
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.act_type == 'mish':
            return x * self.act(x)
        else:
            return self.act(x)


def _add_conv(layers, in_planes, out_planes, kernel_size=3, stride=1, padding=0, groups=1, norm_layer=nn.BatchNorm2d,
              act_type='relu'):
    """

    :param layers:
    :param in_planes:
    :param out_planes:
    :param kernel_size:
    :param stride:
    :param padding:
    :param groups:
    :param norm_layer:
    :param act_type:
    :return:
    """
    layers.append(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False))
    layers.append(norm_layer(out_planes, momentum=0.9, track_running_stats=True))
    layers.append(Activation(act_type))


def _add_conv_dw(layers, in_planes, out_planes, stride, norm_layer=nn.BatchNorm2d, act_type='relu'):
    """

    :param layers:
    :param in_planes:
    :param out_planes:
    :param stride:
    :param norm_layer:
    :param act_type:
    :return:
    """
    _add_conv(layers, in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes,
              norm_layer=norm_layer, act_type=act_type)
    _add_conv(layers, in_planes, out_planes, kernel_size=1, norm_layer=norm_layer, act_type=act_type)


class DetMobileNet(nn.Module):
    def __init__(self, feature_strides=(16, 32), multiplier=1.0, norm_layer=nn.BatchNorm2d, act_type='relu'):
        super(DetMobileNet, self).__init__()
        input_channel = 32
        last_channel = 1024

        features = []
        _add_conv(features, in_planes=3, out_planes=int(input_channel * multiplier), kernel_size=3, stride=2, padding=1,
                  norm_layer=norm_layer, act_type=act_type)
        in_planes = [int(x * multiplier) for x in [32, 64] + [128] * 2 + [256] * 2 + [512] * 6 + [last_channel]]
        out_planes = [int(x * multiplier) for x in [64] + [128] * 2 + [256] * 2 + [512] * 6 + [last_channel] * 2]
        strides = [1, 2] * 3 + [1] * 5 + [2, 1]
        current_stride = 2
        self.ids = []

        for i, (i_p, o_p, s) in enumerate(zip(in_planes, out_planes, strides)):
            if s == 2:
                if current_stride in feature_strides:
                    self.ids.append(i)
                current_stride *= 2
            _add_conv_dw(features, i_p, o_p, s, norm_layer, act_type)
        self.features = nn.ModuleList(features)

    def forward(self, x):
        out = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if (i - 2) / 6 in self.ids:
                out.append(x)
        out.append(x)
        return out


def get_det_mobilenet(feature_strides=(16, 32), multiplier=1.0, norm_layer=nn.BatchNorm2d, act_type='relu'):
    return DetMobileNet(feature_strides, multiplier, norm_layer, act_type)


# if __name__ == '__main__':
#     net = DetMobileNet().cuda(0)
#     # net = nn.Sequential(nn.BatchNorm2d(1024, affine=True, track_running_stats=False),
#     #                     nn.BatchNorm2d(1024)).cuda(0)
#     summary(net, (3, 512, 512))
#     img = torch.rand((1, 3, 512, 512)).cuda()
#     out = net(img)
#     for f in out:
#         print(f.shape)
