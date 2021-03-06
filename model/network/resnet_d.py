"""
# ResNet-D backbone with deep-stem
# Code Adapted from:
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import network.nn.mynn as mynn
from network.nn.coordatt import CoordAtt

__all__ = ['ResNet', 'resnet18', 'resnet50', 'resnet101']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    Basic Block for Resnet
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = mynn.Norm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = mynn.Norm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck Layer for Resnet
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = mynn.Norm2d(planes)

        # dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               bias=False, dilation=dilation, padding=dilation)
        self.bn2 = mynn.Norm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = mynn.Norm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    Resnet
    """

    def __init__(self, block, layers, num_classes=1000):
        # 3??3?????????????????????
        # self.inplanes = 128

        # ??????3??3??????
        self.inplanes = 64

        blocks = [1, 2, 4]
        super(ResNet, self).__init__()
        dilations = [1, 1, 1, 2]
        # dilations = [1, 1, 1, 1]
        strides = [1, 2, 1, 1]

        # ??????AlignNetResNet??????resnet.layer0
        # self.conv1 = nn.Sequential(
        #     conv3x3(3, 64, stride=2),  # ????????????1/2?????????64
        #     mynn.Norm2d(64),
        #     nn.ReLU(inplace=True),
        #
        #     conv3x3(64, 64),  # ?????????????????????64
        #     mynn.Norm2d(64),
        #     nn.ReLU(inplace=True),
        #
        #     conv3x3(64, 128)  # ?????????????????????128
        # )

        # ?????????Aresnet,??????AlignNetResNet??????resnet.layer0
        self.conv1 = nn.Sequential(
            conv3x3(3, 32, stride=2),  # ????????????1/2?????????64
            mynn.Norm2d(32),
            nn.ReLU(inplace=True),

            conv3x3(32, 64),  # ?????????????????????64
            mynn.Norm2d(64),
            nn.ReLU(inplace=True),

            conv3x3(64, 64)  # ?????????????????????64
        )

        # ??????Aresnet???
        # self.bn1 = mynn.Norm2d(128)  # nn.BatchNorm2d(128)
        # ??????Aresnet
        self.bn1 = mynn.Norm2d(64)  # nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1/2

        # ??????AlignNetResNet??????resnet.layer1
        self.layer1 = self._make_layer(block, 64, layers[0], dilation=dilations[0])

        # CA
        # self.ca1 = CoordAtt(64,64)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilation=dilations[1])

        # self.ca2 = CoordAtt(128, 128)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=dilations[2])

        # self.ca3 = CoordAtt(256, 256)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=1, dilation=dilations[3])

        # self.ca4 = CoordAtt(512, 512)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                mynn.Norm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation))
        self.inplanes = planes * block.expansion
        for index in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0] * dilation,
                            downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i] * dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        # x = self.ca1(x)
        x = self.layer2(x)

        # x = self.ca2(x)
        x = self.layer3(x)

        # x = self.ca3(x)
        x = self.layer4(x)

        # x = self.ca4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def _load_pretrained_model(model, url=None, path=None):
    if url is not None:
        pretrain_dict = model_zoo.load_url(url)
    else:
        pretrain_dict = torch.load(path)

    model_dict = {}
    state_dict = model.state_dict()
    for k, v in pretrain_dict.items():
        # print(k)
        if k in state_dict:
            model_dict[k] = v
        else:
            print(k)
    state_dict.update(model_dict)
    model.load_state_dict(state_dict)


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load("./pretrained_models/resnet18-deep-inplane128.pth", map_location="cpu"))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        _load_pretrained_model(model, url='https://download.pytorch.org/models/resnet50-19c8e357.pth')
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        _load_pretrained_model(model, url='https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
    return model


def resnet152(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
