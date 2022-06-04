"""
Custom Norm operators to enable sync BN, regular BN and for weight initialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6), norm_layer=nn.BatchNorm2d):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size, norm_layer) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            norm_layer(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1)
            )

    def _make_stage(self, features, out_features, size, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = norm_layer(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle




#获得具有强语义表示的高分辨率特征图 h_feature
#基于流的对齐模块（FAM）
#
#首先给出相邻两个特征图 F(l)  F(l-1)，首先将特征图 F(l)进行上采样，利用双线性插值，得到与F(l-1)相同尺寸的特征图。
#然后将两个特征图进行concat，再经过一个3X3的卷积，预测出语义流场
#对于空间中的每个位置P(l-1)，通过上采样被映射到点 P(l)
class AlignedModule(nn.Module):

    def __init__(self,inplane, outplane, kernel_size=3):
        super(AlignedModule, self).__init__()

        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make1 = nn.Conv2d(outplane*2, 2, kernel_size=3, padding=1, bias=False)
        self.flow_make2 = nn.Conv2d(outplane * 2, 2, kernel_size=1, padding=0, bias=False)
        self.flow_make3 = nn.Conv2d(outplane * 2, 2, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        low_feature, h_feature = x    #相邻两个特征图F(l-1)  F(l)
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature= self.down_h(h_feature)
        h_feature = F.upsample(h_feature, size=size, mode="bilinear", align_corners=True)#将特征图 F(l)进行上采样,利用双线性插值，得到与F(l-1)相同尺寸的特征图。
        feature = torch.cat([h_feature, low_feature], 1)
        flow1 = self.flow_make1(feature)
        flow2 = self.flow_make2(feature)
        flow3 = self.flow_make3(feature)
        flow = (flow1+flow2+flow3)/3
        #flow = self.flow_make(torch.cat([h_feature, low_feature], 1))#然后将两个特征图进行concat，再经过一个3X3的卷积，预测出语义流场
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)#对于空间中的每个位置P(l-1)，通过上采样被映射到点 P(l)

        return h_feature


#flow_warp
    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm  # 扭曲坐标

        output = F.grid_sample(input, grid)  # 双线性插值得到高分辨率特征
        return output




