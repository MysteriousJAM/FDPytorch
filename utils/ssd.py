import torch

from torch import nn

from utils.mobilenet import get_det_mobilenet
from utils.csposanet import CSPOSANet


class SSD(nn.Module):
    def __init__(self, backbone, trans_filters, backbone_feature_filters, classes=1, anchors_per_location=1,
                 pretrained_backbone=True):
        super(SSD, self).__init__()
        self.trans_filters = trans_filters
        self.in_filters = backbone_feature_filters + trans_filters
        self.classes = classes
        self.anchors_per_location = anchors_per_location

        if backbone == 'mobilenet0.125':
            if pretrained_backbone:
                self.backbone = torch.load('models/mobilenet0.125.pth')
            else:
                self.backbone = get_det_mobilenet(multiplier=0.125)
        elif backbone == 'mobilenet0.25':
            if pretrained_backbone:
                self.backbone = torch.load('models/mobilenet0.25.pth')
            else:    
                self.backbone = CSPOSANet(multiplier=0.125)
        elif backbone == 'csposanet0.125':
            if pretrained_backbone:
                self.backbone = torch.load('models/csposanet0.125.pth')
            else:    
                self.backbone = get_det_mobilenet(multiplier=0.25)
                raise NotImplementedError
        elif backbone == 'csposanet0.25':                                                                                                   
            if pretrained_backbone:
                self.backbone = torch.load('models/csposanet0.25.pth')
            else:
                self.backbone = CSPOSANet(multiplier=0.25)
        else:
            print(backbone)
            raise NotImplementedError

        self.feature_expand = []
        for i in range(len(trans_filters)):
            in_filter = backbone_feature_filters[-1] if i == 0 else trans_filters[i - 1]
            self.feature_expand.append(nn.Sequential(nn.Conv2d(in_channels=in_filter,
                                                               out_channels=trans_filters[i],
                                                               kernel_size=1,
                                                               stride=1,
                                                               padding=0,
                                                               dilation=1,
                                                               groups=1,
                                                               bias=False),
                                                     nn.BatchNorm2d(num_features=trans_filters[i], eps=1e-3,
                                                                    momentum=0.9),
                                                     nn.ReLU(inplace=True),
                                                     nn.Conv2d(in_channels=trans_filters[i],
                                                               out_channels=trans_filters[i],
                                                               kernel_size=3,
                                                               stride=2,
                                                               padding=1,
                                                               dilation=1,
                                                               groups=1,
                                                               bias=False),
                                                     nn.BatchNorm2d(num_features=trans_filters[i], eps=1e-3,
                                                                    momentum=0.9),
                                                     nn.ReLU(inplace=True)
                                                     ))
        self.feature_expand = nn.ModuleList(self.feature_expand)

        # two fors are needed for correct mxnet weights import
        self.class_predictors = []
        self.box_predictors = []
        for i, f in enumerate(self.in_filters):
            self.class_predictors.append(nn.Conv2d(in_channels=f,
                                                   out_channels=classes + 1,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1,
                                                   dilation=1,
                                                   groups=1,
                                                   bias=True))
        for i, f in enumerate(self.in_filters):
            self.box_predictors.append(nn.Conv2d(in_channels=f,
                                                 out_channels=4 * anchors_per_location,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1,
                                                 dilation=1,
                                                 groups=1,
                                                 bias=True))

            self.class_predictors = nn.ModuleList(self.class_predictors)
            self.box_predictors = nn.ModuleList(self.box_predictors)

    def forward(self, x):
        features = self.backbone(x)
        y = features[-1]
        for f in self.feature_expand:
            y = f(y)
            features.append(y)
        out = []
        i = 0
        for feat, cp, bp in zip(features, self.class_predictors, self.box_predictors):
            c = cp(feat).view(feat.size(0), self.classes + 1, -1)
            b = bp(feat).view(feat.size(0), 4, -1)
            out.append((c, b))

        cls_preds, box_preds = list(zip(*out))
        cls_preds, box_preds = torch.cat(cls_preds, 2).contiguous(), torch.cat(box_preds, 2).contiguous()
        return torch.cat([cls_preds, box_preds], dim=1).contiguous()


def get_ssd(backbone, trans_filters, backbone_feature_filters, pretrained_backbone=True):
    return SSD(backbone, trans_filters=trans_filters, backbone_feature_filters=backbone_feature_filters,
               pretrained_backbone=pretrained_backbone)
