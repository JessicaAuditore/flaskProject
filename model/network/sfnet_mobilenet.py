#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Implementation of SFNet ResNet series.
# Author: Xiangtai(lxt@pku.edu.cn)
# Date: 2020/7/1

import torch.nn as nn
import torch

from Lightweight.MobileNetV1 import MobileNetV1
from Lightweight.MobileNetV2 import MobileNetV2
from Lightweight.MobileNetV3 import MobileNetV3
from Lightweight.MobileNetXt import MobileNetXt
from network.nn import JPU
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


from network.nn.coordatt import CoordAtt
from models.aspp import build_aspp

# from nn.customize import JPU, JPU_X

from .nn import SegmentationLosses, SyncBatchNorm


# deconder 解码
class UperNetAlignHead(nn.Module):
    def __init__(self, trunk, num_class, norm_layer=nn.BatchNorm2d, fpn_inplanes=None, fpn_dim=None,
                 conv3x3_type="conv", fpn_dsn=False):
        super(UperNetAlignHead, self).__init__()
        self.jpu=False
        if fpn_inplanes is None:
            fpn_inplanes = [128, 256, 512, 1024]
        if fpn_inplanes is None:
            fpn_dim = 128
        if trunk == 'resnet50':
            # (256,512,1024,2048)
            self.jpu = JPU([512, 1024, 2048], width=512, norm_layer=nn.BatchNorm2d, up_kwargs=up_kwargs)
            fpn_inplanes = [256, 512, 1024, 2048]
            fpn_dim = 256
        if trunk == 'mobilenetv1':
            self.aspp = build_aspp(trunk, 32, nn.BatchNorm2d)
            fpn_inplanes = [128, 256, 512, 1024]
            fpn_dim = 128
        elif trunk == 'mobilenetv2':
            self.aspp = build_aspp(trunk, 32, nn.BatchNorm2d)
            fpn_inplanes = [24, 32, 96, 1280]
            fpn_dim = 24
        elif trunk == 'mobilenetv3':
            self.aspp = build_aspp(trunk, 32, nn.BatchNorm2d)
            fpn_inplanes = [24, 80, 112, 1280]
            fpn_dim = 24
        elif trunk == 'mobilenetxt':
            self.jpu = JPU([192, 384, 1280], width=192, norm_layer=nn.BatchNorm2d, up_kwargs=up_kwargs)
            # self.aspp = build_aspp(trunk, 32, nn.BatchNorm2d)
            # 有JPU
            fpn_inplanes = [144, 192, 384, 768]
            # 无JPU
            # fpn_inplanes = [144, 192, 384, 1280]
            fpn_dim = 144
        else:
            print("wrong trunk")

        # 去除ASPP的对A5的处理
        self.conv5 = nn.Sequential(
            nn.Conv2d(fpn_inplanes[-1], fpn_dim, 1),
            norm_layer(fpn_dim),
            nn.ReLU(inplace=True),
        )

        # self.fpn_dsn = fpn_dsn
        self.fpn_in = []  # 改变通道数的1×1的卷积核，特征图输入到解码模块特征融合时需要先进行卷积改变图像通道
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    norm_layer(fpn_dim),
                    nn.ReLU(inplace=False)
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)  # 改变通道数的1×1的卷积核

        self.fpn_out = []  # 三个3×3的卷积核
        self.fpn_out_align = []
        # self.dsn = []
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
            print(fpn_dim)
            if conv3x3_type == 'conv':
                self.fpn_out_align.append(
                    AlignedModule(inplane=fpn_dim, outplane=fpn_dim // 2)
                )

        self.fpn_out = nn.ModuleList(self.fpn_out)  # 三个3×3的卷积核
        self.fpn_out_align = nn.ModuleList(self.fpn_out_align)  # 流对齐模块

        # if self.fpn_dsn:
        #     self.dsn = nn.ModuleList(self.dsn)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out):  # conv_out=[x1,x2,x3,x4]
        if self.jpu:
            jpu_out = self.jpu(conv_out[0], conv_out[1], conv_out[2], conv_out[3])
            # 元组tuple转list列表
            jpu_out = list(jpu_out)

            jpu_out[-1] = self.conv5(jpu_out[-1])
            f = jpu_out[-1]
            fpn_feature_list = [jpu_out[-1]]
        else:
            # aspp_out = self.aspp(conv_out[-1])  # 得到将x4进行空洞卷积后的结果
            # 无ASPP
            aspp_out = self.conv5(conv_out[-1])
            f = aspp_out
            fpn_feature_list = [aspp_out]  # p5
        out = []
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)  # lateral branch，1×1卷积
            f = self.fpn_out_align[i]([conv_x, f])  # f表示上一层的特征图（即更多通道更深层次的特征图）放大两倍，
            # 将两倍的上一层次特征图和本层次特征图进行流对齐并逐像素相加

            f = conv_x + f  # 两层特征图相加
            fpn_feature_list.append(self.fpn_out[i](f))  # fpn_out是3*3的卷积改变减少线性插值的影响，[P4 - P2]

            # if self.fpn_dsn:
            #     out.append(self.dsn[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]  # 输出图片的尺寸(128,128)
        fusion_list = [fpn_feature_list[0]]  # p2:(8,256,128,128)

        for i in range(1, len(fpn_feature_list)):  # 将其余特征图p2,p3,p4,p5尺寸放大到128*128
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=True))

        fusion_out = torch.cat(fusion_list, 1)  # 在通道维度上拼接四个特征图
        x = self.conv_last(fusion_out)  # 改变通道数1024->256,256->2

        return x, out  # out为[]


up_kwargs = {'mode': 'bilinear', 'align_corners': True}


class AlignNetResNet(nn.Module):
    def __init__(self, num_classes, trunk='mobilenetv1', variant='D',
                 skip='m1', skip_num=48, fpn_dsn=False, jpu=None):
        super(AlignNetResNet, self).__init__()
        self.resnet50 = False
        # enconder
        if trunk == 'resnet50':
            self.resnet50 = True
            resnet = resnet50()
            # Resnet——model
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                                   padding=3, bias=False)  # padding填充一般为卷积核大小的一半
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.layer1 = resnet.layer1
            #         self.ca1 = CoordAtt(256, 256)
            self.layer2 = resnet.layer2
            #         self.ca2 = CoordAtt(512, 512)
            self.layer3 = resnet.layer3
            # self.ca3 = CoordAtt(1024, 1024)
            self.layer4 = resnet.layer4
            # self.ca4 = CoordAtt(2048, 2048)
            del resnet
        if trunk == 'mobilenetv1':
            self.mobilenet = MobileNetV1()
        elif trunk == 'mobilenetv2':
            self.mobilenet = MobileNetV2()
        elif trunk == 'mobilenetv3':
            self.mobilenet = MobileNetV3(type='large')
        elif trunk == 'mobilenetxt':
            self.mobilenet = MobileNetXt()
        else:
            print("wrong trunk")

        # deconder
        self.head = UperNetAlignHead(trunk, num_class=num_classes, norm_layer=Norm2d, fpn_dsn=fpn_dsn)

    def forward(self, x, gts=None):
        x_size = x.size()
        if self.resnet50:
            x0 = self.conv1(x)
            x0 = self.bn1(x0)
            x0 = self.relu(x0)
            x0 = self.maxpool(x0)

            x1 = self.layer1(x0)  # (8,256,128,128)
            #         x1 = self.ca1(x1)
            x2 = self.layer2(x1)  # (8,512,64,64)
            # x2 = self.ca2(x2)
            x3 = self.layer3(x2)  # (8,1024,32,32)
            # x3 = self.ca3(x3)
            x4 = self.layer4(x3)  # (8,2048,16,16)
            # x4 = self.ca4(x4)
            x_list = [x1, x2, x3, x4]  # (256,512,1024,2048)
        else:
            x_list = self.mobilenet(x)
        x = self.head(x_list)
        main_out = Upsample(x[0], x_size[2:])

        return main_out


def mobilenetv1_SF_deeply(num_classes):
    """
    ResNet-50 Based Network
    """
    return AlignNetResNet(num_classes, trunk='mobilenetv1', variant='D', skip='m1')


def mobilenetv2_SF_deeply(num_classes):
    """
    ResNet-50 Based Network
    """
    return AlignNetResNet(num_classes, trunk='mobilenetv2', variant='D', skip='m1')


def mobilenetv3_SF_deeply(num_classes):
    """
    ResNet-50 Based Network
    """
    return AlignNetResNet(num_classes, trunk='mobilenetv3', variant='D', skip='m1')


def mobilenetxt_SF_deeply(num_classes):
    """
    ResNet-50 Based Network
    """
    return AlignNetResNet(num_classes, trunk='mobilenetxt', variant='D', skip='m1', jpu='JPU')


def resnet50_SF_deeply(num_classes):
    """
    ResNet-50 Based Network
    """
    return AlignNetResNet(num_classes, trunk='resnet50', variant='D', skip='m1', jpu='JPU')
