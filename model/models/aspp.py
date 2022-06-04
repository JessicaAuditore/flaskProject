import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        outplane = 256
        if backbone == 'mobilenetv1':
            inplanes = 1024
            outplane = 128
        elif backbone == 'mobilenetv2':
            inplanes = 1280
            outplane = 24
        elif backbone == 'mobilenetv3':
            inplanes = 1280
            outplane = 24
        elif backbone == 'mobilenetxt':
            inplanes = 1280
            outplane = 144
        elif backbone == 'mobilenetxtjpu':
            inplanes = 768
            outplane = 192
        elif backbone == 'mobilenetxtnojpu':
            inplanes = 1280
            outplane = 192
        else:
            inplanes = 2048
        if output_stride == 32:
            # dilations = [1, 6, 12, 18]
#             dilations = [1, 2, 4, 8]
#             dilations = [1, 2, 4, 8]
            dilations = [1, 3, 6, 9]
            # dilations = [1, 4, 8, 12]
        elif output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError
        self.aspp1 = _ASPPModule(inplanes, outplane, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, outplane, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, outplane, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, outplane, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),  # 自适应平均池化，输出图像的H和W为(1，1)
                                             nn.Conv2d(inplanes, outplane, 1, stride=1, bias=False),  # 改变通道数
                                             BatchNorm(outplane),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(outplane*5, outplane, 1, bias=False)
        self.bn1 = BatchNorm(outplane)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)  # 通道：2048->256，尺寸不变
        x2 = self.aspp2(x)  # 通道：2048->256，尺寸不变
        x3 = self.aspp3(x)  # 通道：2048->256，尺寸不变
        x4 = self.aspp4(x)  # 通道：2048->256,尺寸不变
        x5 = self.global_avg_pool(x)    # 通道：2048->256,尺寸变成1*1
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)   # 插值法恢复尺寸，选取x4的宽和高
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # 在通道维度上进行拼接，即256*5=1280

        x = self.conv1(x)   # 通道：1280->256
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)   # 随机丢弃

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_aspp(backbone, output_stride, BatchNorm):
    return ASPP(backbone, output_stride, BatchNorm)