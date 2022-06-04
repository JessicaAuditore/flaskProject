import torch
import torch.nn as nn
import torch.nn.functional as F
from model.network.nn.encoding import Encoding
from model.Lightweight.MobileNetV1 import MobileNetV1
from model.Lightweight.MobileNetV2 import MobileNetV2
from model.Lightweight.MobileNetV3 import MobileNetV3
from model.Lightweight.MobileNetXt import MobileNetXt
from .fcn import FCNHead
from .nn.customize import JPU, JPU_X, Mean
from .nn import SyncBatchNorm



up_kwargs = {'mode': 'bilinear', 'align_corners': True}


from model.models.aspp import build_aspp


class EncNet(nn.Module):
    def __init__(self, nclass, trunk=None, jpu=None, aspp=False, aux=False, se_loss=False,
                 norm_layer=SyncBatchNorm, **kwargs):
        super(EncNet, self).__init__()
        self.resnet50 = False
        self.is_aspp = aspp
        # enconder
        if trunk == 'mobilenetv1':
            self.mobilenet = MobileNetV1()
            if jpu == 'JPU':
                self.jpu = JPU([256, 512, 1024], width=256, norm_layer=SyncBatchNorm, up_kwargs=up_kwargs)
            elif jpu == 'JPU_X':
                self.jpu = JPU_X([256, 512, 1024], width=256, norm_layer=SyncBatchNorm, up_kwargs=up_kwargs)
            self.head = EncHead([256, 512, 1024], nclass, se_loss=se_loss, jpu=jpu,
                                lateral=True, norm_layer=norm_layer,
                                up_kwargs=up_kwargs)
        elif trunk == 'mobilenetv2':
            self.mobilenet = MobileNetV2()
            if jpu == 'JPU':
                self.jpu = JPU([32, 96, 1280], width=32, norm_layer=SyncBatchNorm, up_kwargs=up_kwargs)
            elif jpu == 'JPU_X':
                self.jpu = JPU_X([32, 96, 1280], width=32, norm_layer=SyncBatchNorm, up_kwargs=up_kwargs)
            self.head = EncHead([32, 96, 128], nclass, se_loss=se_loss, jpu=jpu,
                                lateral=False, norm_layer=norm_layer,
                                up_kwargs=up_kwargs)
        elif trunk == 'mobilenetv3':
            self.mobilenet = MobileNetV3(type='large')

            if jpu == 'JPU':
                self.jpu = JPU([80, 112, 1280], width=80, norm_layer=SyncBatchNorm, up_kwargs=up_kwargs)
            elif jpu == 'JPU_X':
                self.jpu = JPU_X([80, 112, 1280], width=80, norm_layer=SyncBatchNorm, up_kwargs=up_kwargs)

            self.head = EncHead([80, 112, 1280], nclass, se_loss=se_loss, jpu=jpu,
                                lateral=True, norm_layer=norm_layer,
                                up_kwargs=up_kwargs)
        elif trunk == 'mobilenetxt':
            self.mobilenet = MobileNetXt()
            if jpu == 'JPU':
                self.jpu = JPU([192, 384, 1280], width=192, norm_layer=SyncBatchNorm, up_kwargs=up_kwargs)
            elif jpu == 'JPU_X':
                self.jpu = JPU_X([192, 384, 1280], width=192, norm_layer=SyncBatchNorm, up_kwargs=up_kwargs)
            if aspp:
                self.aspp = build_aspp(trunk + 'jpu', 32, nn.BatchNorm2d)
                self.head = EncHead([192, 384, 192], nclass, se_loss=se_loss, jpu=jpu,
                                    lateral=True, norm_layer=norm_layer,
                                    up_kwargs=up_kwargs)
            else:
                self.head = EncHead([192, 384, 768], nclass, se_loss=se_loss, jpu=jpu,
                                    lateral=True, norm_layer=norm_layer,
                                    up_kwargs=up_kwargs)
        elif trunk == 'resnet50':
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
            self.jpu = JPU([512, 1024, 2048], width=512, norm_layer=SyncBatchNorm, up_kwargs=up_kwargs)
            self.head = EncHead([512, 1024, 2048], nclass, se_loss=se_loss, jpu=jpu,
                                lateral=True, norm_layer=norm_layer,
                                up_kwargs=up_kwargs)
        else:
            print("wrong trunk")
        self.aux = aux
        if self.aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer=norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        # mobilenetxt
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
            x_list = [x1, x2, x3, x4]
        else:
            x_list = self.mobilenet(x)
        # jpu
        features = self.jpu(x_list[0], x_list[1], x_list[2], x_list[3])
        # aspp
        if self.is_aspp:
            features = list(features)
            features[3] = self.aspp(features[3])
        x = self.head(*features)
        x = F.interpolate(x, imsize, **up_kwargs)
        return x


from model.network.model import resnet50


class EncNet_nojpu(nn.Module):
    def __init__(self, nclass, trunk=None, jpu=None, aspp=False, aux=False, se_loss=False,
                 norm_layer=SyncBatchNorm, **kwargs):
        super(EncNet_nojpu, self).__init__()
        self.aspp = False
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
            if aspp:
                self.aspp = build_aspp(trunk + 'jpu', 32, nn.BatchNorm2d)
                self.head = EncHead([512, 1024, 512], nclass, se_loss=se_loss, jpu=jpu,
                                    lateral=True, norm_layer=norm_layer,
                                    up_kwargs=up_kwargs)
            else:
                self.head = EncHead([512, 1024, 2048], nclass, se_loss=se_loss, jpu=jpu,
                                    lateral=True, norm_layer=norm_layer,
                                    up_kwargs=up_kwargs)
        if trunk == 'mobilenetv1':
            self.mobilenet = MobileNetV1()
        elif trunk == 'mobilenetv2':
            self.mobilenet = MobileNetV2()
            if aspp:
                self.aspp = build_aspp(trunk + 'jpu', 32, nn.BatchNorm2d)
                self.head = EncHead([32, 96, 1280], nclass, se_loss=se_loss, jpu=jpu,
                                    lateral=True, norm_layer=norm_layer,
                                    up_kwargs=up_kwargs)
            else:
                self.head = EncHead([32, 96, 1280], nclass, se_loss=se_loss, jpu=jpu,
                                    lateral=True, norm_layer=norm_layer,
                                    up_kwargs=up_kwargs)
        elif trunk == 'mobilenetv3':
            self.mobilenet = MobileNetV3(type='large')

            self.head = EncHead([80, 112, 1280], nclass, se_loss=se_loss, jpu=jpu,
                                lateral=True, norm_layer=norm_layer,
                                up_kwargs=up_kwargs)
        elif trunk == 'mobilenetxt':
            self.mobilenet = MobileNetXt()
            if aspp:
                self.aspp = build_aspp(trunk + 'nojpu', 32, nn.BatchNorm2d)
                self.head = EncHead([192, 384, 192], nclass, se_loss=se_loss, jpu=jpu,
                                    lateral=True, norm_layer=norm_layer,
                                    up_kwargs=up_kwargs)
            else:
                self.head = EncHead([192, 384, 1280], nclass, se_loss=se_loss, jpu=jpu,
                                    lateral=True, norm_layer=norm_layer,
                                    up_kwargs=up_kwargs)
        else:
            print("wrong trunk")

    def forward(self, x):
        imsize = x.size()[2:]
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
            features = [x1, x2, x3, x4]
        else:
            # mobilenetxt
            x_list = self.mobilenet(x)
            features = x_list
            # aspp
        if self.aspp:
            features = list(features)
            features[3] = self.aspp(features[3])
        x2_size = features[1].size()[2:]
        features[3] = F.interpolate(features[3], x2_size, **up_kwargs)

        x = self.head(*features)
        x = F.interpolate(x, imsize, **up_kwargs)
        print('0')
        return x


class EncModule(nn.Module):
    def __init__(self, in_channels, nclass, ncodes=32, se_loss=True, norm_layer=None):
        super(EncModule, self).__init__()
        self.se_loss = se_loss
        self.encoding = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            norm_layer(in_channels),
            nn.ReLU(inplace=True),
            Encoding(D=in_channels, K=ncodes),
            norm_layer(ncodes),
            nn.ReLU(inplace=True),
            Mean(dim=1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid())
        if self.se_loss:
            self.selayer = nn.Linear(in_channels, nclass)

    def forward(self, x):
        en = self.encoding(x)
        b, c, _, _ = x.size()
        gamma = self.fc(en)
        y = gamma.view(b, c, 1, 1)
        outputs = [F.relu_(x + x * y)]
        if self.se_loss:
            outputs.append(self.selayer(en))
        return outputs


class EncHead(nn.Module):
    def __init__(self, in_channels, out_channels, se_loss=True, jpu=True, lateral=False,
                 norm_layer=None, up_kwargs=None):
        super(EncHead, self).__init__()
        self.se_loss = se_loss
        self.lateral = lateral
        self.up_kwargs = up_kwargs
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels[-1], 192, 1, bias=False),
                                   norm_layer(192),
                                   nn.ReLU(inplace=True)) if jpu else \
            nn.Sequential(nn.Conv2d(in_channels[-1], 192, 3, padding=1, bias=False),
                          norm_layer(192),
                          nn.ReLU(inplace=True))
        if lateral:
            self.connect = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_channels[0], 192, kernel_size=1, bias=False),
                    norm_layer(192),
                    nn.ReLU(inplace=True)),
                nn.Sequential(
                    nn.Conv2d(in_channels[1], 192, kernel_size=1, bias=False),
                    norm_layer(192),
                    nn.ReLU(inplace=True)),
            ])
            self.fusion = nn.Sequential(
                nn.Conv2d(3 * 192, 192, kernel_size=3, padding=1, bias=False),
                norm_layer(192),
                nn.ReLU(inplace=True))
        self.encmodule = EncModule(192, out_channels, ncodes=32,
                                   se_loss=se_loss, norm_layer=norm_layer)
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False),
                                   nn.Conv2d(192, out_channels, 1))

    def forward(self, *inputs):
        feat = self.conv5(inputs[-1])
        if self.lateral:
            c2 = self.connect[0](inputs[1])
            # 改变c3尺寸
            _, _, h, w = c2.size()
            c3 = self.connect[1](inputs[2])
            c3 = F.interpolate(c3, (h, w), mode='bilinear', align_corners=True)

            feat = self.fusion(torch.cat([feat, c2, c3], 1))
        # 有编码
        outs = self.encmodule(feat)
        # 无编码
        # outs = [feat]

        outs = self.conv6(outs[0])
        return outs


def mobilenetv1_EncNet(num_classes, jpu=None, aspp=False):
    """
    ResNet-50 Based Network
    """
    return EncNet(num_classes, trunk='mobilenetv1', jpu=jpu, aspp=aspp)


def mobilenetv2_EncNet(num_classes, jpu=None, aspp=False):
    """
    ResNet-50 Based Network
    """
    return EncNet(num_classes, trunk='mobilenetv2', jpu=jpu, aspp=aspp)


def mobilenetv3_EncNet(num_classes, jpu=None, aspp=False):
    """
    ResNet-50 Based Network
    """
    return EncNet(num_classes, trunk='mobilenetv3', jpu=jpu, aspp=aspp)


def mobilenetxt_EncNet(num_classes, jpu=None, aspp=False):
    """
    ResNet-50 Based Network
    """
    return EncNet(num_classes, trunk='mobilenetxt', jpu=jpu, aspp=aspp)


def Resnet_EncNet(num_classes, jpu=None, aspp=False):
    """
    ResNet-50 Based Network
    """
    return EncNet(num_classes, trunk='resnet50', jpu=jpu, aspp=aspp)


def xt_EncNet_nojpu(num_classes, jpu=None, aspp=False):
    """
    ResNet-50 Based Network
    """
    return EncNet_nojpu(num_classes, trunk='mobilenetxt', jpu=jpu, aspp=aspp)


def v2_EncNet_nojpu(num_classes, jpu=None, aspp=False):
    """
    ResNet-50 Based Network
    """
    return EncNet_nojpu(num_classes, trunk='mobilenetv2', jpu=jpu, aspp=aspp)


def Resnet_EncNet_nojpu(num_classes, jpu=None, aspp=False):
    """
    ResNet-50 Based Network
    """
    return EncNet_nojpu(num_classes, trunk='resnet50', jpu=jpu, aspp=aspp)
