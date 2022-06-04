import torch.nn as nn


def BottleneckV1(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1,
                  groups=in_channels),
        nn.BatchNorm2d(in_channels),
        nn.ReLU6(inplace=True),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()

        # 输入为1，输出为1/2
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        # 输入为1/2，输出为1/4
        self.bottleneck1 = nn.Sequential(
            # 1/2
            BottleneckV1(32, 64, stride=1),
            BottleneckV1(64, 128, stride=2),
        )

        # 输入为1/4，输出为1/8
        self.bottleneck2 = nn.Sequential(
            # 1/4
            BottleneckV1(128, 128, stride=1),
            BottleneckV1(128, 256, stride=2),
        )

        # 输入为1/8，输出为1/16
        self.bottleneck3 = nn.Sequential(
            # 1/8
            BottleneckV1(256, 256, stride=1),
            BottleneckV1(256, 512, stride=2),
        )

        # 输入为1/16，输出为1/32
        self.bottleneck4 = nn.Sequential(
            # 1/16
            BottleneckV1(512, 512, stride=1),
            BottleneckV1(512, 512, stride=1),
            BottleneckV1(512, 512, stride=1),
            BottleneckV1(512, 512, stride=1),
            BottleneckV1(512, 512, stride=1),
            BottleneckV1(512, 1024, stride=2),
        )

        # 输入为1/32，输出为1/32
        self.bottleneck5 = nn.Sequential(
            # 1/32
            BottleneckV1(1024, 1024, stride=1),
        )

        # self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        # self.linear = nn.Linear(in_features=1024, out_features=num_classes)
        # self.dropout = nn.Dropout(p=0.2)
        # self.softmax = nn.Softmax(dim=1)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)  # 输出为1/2
        x1 = self.bottleneck1(x)  # 输出为1/4
        x2 = self.bottleneck2(x1)  # 输出为1/8
        x3 = self.bottleneck3(x2)  # 输出为1/16
        x4 = self.bottleneck4(x3)  # 输出为1/32
        x5 = self.bottleneck5(x4)  # 输出为1/32
        # x = self.avg_pool(x5)
        # x = x.view(x.size(0), -1)
        # x = self.dropout(x)
        # x = self.linear(x)
        # out = self.softmax(x)
        return [x1, x2, x3, x5]
