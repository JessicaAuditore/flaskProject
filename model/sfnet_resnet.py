import torch.nn as nn
import torch
from model.network.nn.operators import AlignedModule
from model.resnet_d import resnet50, resnet101
from model.network.nn.mynn import Norm2d, Upsample
from model.network.nn.coordatt import CoordAtt
from model.models.aspp import build_aspp


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3_bn_relu(in_planes, out_planes, stride=1, normal_layer=nn.BatchNorm2d):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        normal_layer(out_planes),
        nn.ReLU(inplace=True),
    )


class UperNetAlignHead(nn.Module):
    def __init__(self, inplane, num_class, norm_layer=nn.BatchNorm2d, fpn_inplanes=[256, 512, 1024, 2048], fpn_dim=256,
                 conv3x3_type="conv", fpn_dsn=False):
        super(UperNetAlignHead, self).__init__()

        self.aspp = build_aspp(inplane, 32, nn.BatchNorm2d)

        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    norm_layer(fpn_dim),
                    nn.ReLU(inplace=False)
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        self.fpn_out_align = []
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
            if conv3x3_type == 'conv':
                self.fpn_out_align.append(
                    AlignedModule(inplane=fpn_dim, outplane=fpn_dim // 2)
                )

        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.fpn_out_align = nn.ModuleList(self.fpn_out_align)
        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out):
        aspp_out = self.aspp(conv_out[-1])

        f = aspp_out
        # fpn_feature_list = [psp_out]
        fpn_feature_list = [aspp_out]  # p5
        out = []
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)
            f = self.fpn_out_align[i]([conv_x, f])

            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse()
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]

        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=True))

        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        return x, out


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
        else:
            raise ValueError("Not a valid network arch")
        resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer0 = resnet.layer0

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.ca3 = CoordAtt(1024, 1024)
        self.layer4 = resnet.layer4
        self.ca4 = CoordAtt(2048, 2048)
        del resnet
        if self.variant == 'D':
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding = (2, 2), (2, 2)
                elif 'downsample.0' in n:
                    m.stride = (2, 2)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding = (4, 4), (4, 4)
        else:
            print("Not using Dilation ")

        inplane_head = 256
        self.head = UperNetAlignHead(inplane_head, num_class=num_classes, norm_layer=Norm2d, fpn_dsn=fpn_dsn)

    def forward(self, x, gts=None):
        x_size = x.size()
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x3 = self.ca3(x3)
        x4 = self.layer4(x3)
        x4 = self.ca4(x4)
        x = self.head([x1, x2, x3, x4])
        main_out = Upsample(x[0], x_size[2:])

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
