#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Implementation of SFNet ResNet series.
# Author: Xiangtai(lxt@pku.edu.cn)
# Date: 2020/7/1

import torch.nn as nn
import torch
from network.nn.operators import AlignedModule, PSPModule

# from network.resnet_d import resnet50, resnet101
from network.model import resnet50, resnet101
from network.nn.mynn import Norm2d, Upsample


# Conv2d3x3
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# Conv2d3x3  batchnorm2d  ReLU
def conv3x3_bn_relu(in_planes, out_planes, stride=1, normal_layer=nn.BatchNorm2d):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        normal_layer(out_planes),
        nn.ReLU(inplace=True),
    )


# deconder
class UperNetAlignHead(nn.Module):
    def __init__(self, inplane, num_class, norm_layer=nn.BatchNorm2d, fpn_inplanes=[256, 512, 1024, 2048], fpn_dim=256,
                 conv3x3_type="conv", fpn_dsn=False):
        super(UperNetAlignHead, self).__init__()
        # ppm
        # self.ppm = PSPModule(inplane, norm_layer=norm_layer, out_features=fpn_dim)

        self.aspp = build_aspp(inplane, 16, nn.BatchNorm2d)

        # 去除ASPP的对A5的处理
        self.conv5 = nn.Sequential(
            nn.Conv2d(2048, 256, 1),
            norm_layer(256),
            nn.ReLU(inplace=True),
        )

        self.fpn_dsn = fpn_dsn
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    norm_layer(fpn_dim),
                    nn.ReLU(inplace=False)
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)    # 改变通道数的1×1的卷积核

        self.fpn_out = []
        self.fpn_out_align = []
        self.dsn = []
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
            print(fpn_dim)
            if conv3x3_type == 'conv':
                self.fpn_out_align.append(
                    AlignedModule(inplane=fpn_dim, outplane=fpn_dim // 2)
                )

            if self.fpn_dsn:
                self.dsn.append(
                    nn.Sequential(
                        nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1),
                        norm_layer(fpn_dim),
                        nn.ReLU(),
                        nn.Dropout2d(0.1),
                        nn.Conv2d(fpn_dim, num_class, kernel_size=1, stride=1, padding=0, bias=True)
                    )
                )

        self.fpn_out = nn.ModuleList(self.fpn_out)  # 三个3×3的卷积核
        self.fpn_out_align = nn.ModuleList(self.fpn_out_align)  # 流对齐模块

        if self.fpn_dsn:
            self.dsn = nn.ModuleList(self.dsn)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out):
        # psp_out = self.ppm(conv_out[-1])
        # aspp_out = self.aspp(conv_out[-1])   # 得到将x4进行空洞卷积后的结果
        # f = psp_out
        aspp_out=self.conv5(conv_out[-1])
        f = aspp_out
        # fpn_feature_list = [psp_out]
        fpn_feature_list = [aspp_out]   # p5
        out = []
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)  # lateral branch
            f = self.fpn_out_align[i]([conv_x, f])   # f表示上一层的特征图（即更多通道更深层次的特征图）放大两倍，
                                                     # 将两倍的上一层次特征图和本层次特征图进行流对齐并逐像素相加

            f = conv_x + f  # 两层特征图相加
            fpn_feature_list.append(self.fpn_out[i](f))     # fpn_out是3*3的卷积改变减少线性插值的影响，[P4 - P2]

            if self.fpn_dsn:
                out.append(self.dsn[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]  # 输出图片的尺寸(128,128)
        fusion_list = [fpn_feature_list[0]]    # p2:(8,256,128,128)

        for i in range(1, len(fpn_feature_list)):  # 将其余特征图p2,p3,p4,p5尺寸放大到128*128
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=True))

        fusion_out = torch.cat(fusion_list, 1)  # 在通道维度上拼接四个特征图
        x = self.conv_last(fusion_out)      # 改变通道数1024->256,256->2

        return x, out   # out为[]

from network.nn.coordatt import CoordAtt
from models.aspp import build_aspp


class AlignNetResNet(nn.Module):
    def __init__(self, num_classes, trunk='resnet50', variant='D',
                 skip='m1', skip_num=48, fpn_dsn=False):
        super(AlignNetResNet, self).__init__()

        self.variant = variant
        self.skip = skip
        self.skip_num = skip_num
        self.fpn_dsn = fpn_dsn

        if trunk == trunk == 'resnet50':
            resnet = resnet50()
        elif trunk == 'resnet101':
            resnet = resnet101()
        # # elif trunk == 'resnet18':
        #     resnet = resnet18()
        else:
            raise ValueError("Not a valid network arch")
        # enconder--Aresnet
        # resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        # self.layer0 = resnet.layer0

        # Resnet——model
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)  # padding填充一般为卷积核大小的一半
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = resnet.layer1
        self.ca1 = CoordAtt(256, 256)
        self.layer2 = resnet.layer2
        self.ca2 = CoordAtt(512, 512)
        self.layer3 = resnet.layer3
        self.ca3 = CoordAtt(1024, 1024)
        self.layer4 = resnet.layer4
        self.ca4 = CoordAtt(2048, 2048)
        self.aspp = build_aspp(resnet, 16, nn.BatchNorm2d)
        self.conv_aspp = nn.Conv2d(256, 2048, kernel_size=1)
        del resnet
        # Dilation
        if self.variant == 'D':
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding = (2, 2), (2, 2)
                elif 'downsample.0' in n:
                    m.stride = (2, 2)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding = (4, 4), (4, 4)
                elif 'downsample.0' in n:
                    m.stride = (2, 2)
        else:
            print("Not using Dilation ")

        inplane_head = 256

        # deconder
        self.head = UperNetAlignHead(inplane_head, num_class=num_classes, norm_layer=Norm2d, fpn_dsn=fpn_dsn)

    def forward(self, x, gts=None):
        x_size = x.size()  # (8,3,512,512)

        # Aresnet——resnet_d
        # x0 = self.layer0(x)  # (8,128,128,128)

        # Resnet——model文件导入
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)


        x0 = self.maxpool(x0)  # 特征图分辨率降低为1/4，通道数仍然为128
        x1 = self.layer1(x0)  # (8,256,128,128)
        #         x1 = self.ca1(x1)
        x2 = self.layer2(x1)  # (8,512,64,64)
        #         x2 = self.ca2(x2)
        x3 = self.layer3(x2)  # (8,1024,32,32)
        # x3 = self.ca3(x3)
        x4 = self.layer4(x3)  # (8,2048,16,16)
        # x4 = self.ca4(x4)
        xa = self.aspp(x4)
        # xa = self.conv_aspp(xa)

        x = self.head([x1, x2, x3, x4])  # (8,2,128,128)
        main_out = Upsample(x[0], x_size[2:])   # 上采样：图片尺寸128->512

        return main_out


def DeepR101_SF_deeply(num_classes):
    """
    ResNet-50 Based Network
    """
    return AlignNetResNet(num_classes, trunk='resnet101', variant='D', skip='m1')


def DeepR50_SF_deeply(num_classes):
    """
    ResNet-50 Based Network
    """
    return AlignNetResNet(num_classes, trunk='resnet50', variant='D', skip='m1')


def DeepR18_SF_deeply(num_classes, criterion):
    """
    ResNet-18 Based Network
    """
    return AlignNetResNet(num_classes, trunk='resnet-18-deep', criterion=criterion, variant='D', skip='m1')


def DeepR18_SF_deeply_dsn(num_classes, criterion):
    """
    ResNet-18 Based Network wtih DSN supervision
    """
    return AlignNetResNet(num_classes, trunk='resnet-18-deep', criterion=criterion, variant='D', skip='m1',
                          fpn_dsn=True)
