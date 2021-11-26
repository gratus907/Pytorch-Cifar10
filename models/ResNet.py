import torch.nn as nn

cfg = {
    'resnet20-cifar' : [3, 3, 3],
    'resnet32-cifar' : [5, 5, 5],
    'resnet44-cifar' : [7, 7, 7],
    'resnet56-cifar' : [9, 9, 9],
    'resnet-orig50' : [4, 6, 3],
    'resnet-orig101' : [4, 23, 3],
    'resnet-orig152' : [8, 36, 3]
}

# Resnetv1.5 fix implemented
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=(1, 1), mode="regular"):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)

        if mode == 'downsample':
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias = False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.downsample = None

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        residual = x
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(output + residual)

class BottleneckResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=(1, 1), mode="regular"):
        super(BottleneckResBlock, self).__init__()
        self.bottleneck_1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch//4, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
            nn.BatchNorm2d(out_ch//4),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch//4, out_ch//4, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_ch//4),
            nn.ReLU()
        )
        self.bottleneck_2 = nn.Sequential(
            nn.Conv2d(out_ch//4, out_ch, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_ch)
        )
        self.relu = nn.ReLU()

        if mode == 'downsample':
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias = False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.downsample = None

    def forward(self, x):
        output = self.bottleneck_1(x)
        output = self.conv(output)
        output = self.bottleneck_2(output)
        residual = x
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(output + residual)

class ResNet(nn.Module):
    def __init__(self, config = 'resnet20-cifar', num_classes=10):
        super(ResNet, self).__init__()
        self.name = config
        num_blocks = cfg[config]
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(16)
        )

        self.layer2 = [ResBlock(16, 16, (1, 1), mode='regular')]
        for i in range(num_blocks[0] - 1):
            self.layer2.append(ResBlock(16, 16, (1, 1), mode='regular'))
        self.layer2 = nn.ModuleList(self.layer2)

        self.layer3 = [ResBlock(16, 32, (2, 2), mode='downsample')]
        for i in range(num_blocks[1] - 1):
            self.layer3.append(ResBlock(32, 32, (1, 1), mode='regular'))
        self.layer3 = nn.ModuleList(self.layer3)

        self.layer4 = [ResBlock(32, 64, (2, 2), mode='downsample')]
        for i in range(num_blocks[2] - 1):
            self.layer4.append(ResBlock(64, 64, (1, 1), mode='regular'))
        self.layer4 = nn.ModuleList(self.layer4)

        self.avgpool = nn.AvgPool2d(kernel_size = (8, 8))
        self.fc = nn.Linear(64, num_classes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        for i in range(len(self.layer2)):
            out = self.layer2[i](out)
        for i in range(len(self.layer3)):
            out = self.layer3[i](out)
        for i in range(len(self.layer4)):
            out = self.layer4[i](out)
        out = self.avgpool(out)
        out = nn.Flatten()(out)
        out = self.fc(out)
        return out

class LargeResNet(nn.Module):
    def __init__(self, config = 'resnet50', num_classes=10):
        super(LargeResNet, self).__init__()
        self.name = config
        num_blocks = cfg[config]
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            BottleneckResBlock(64, 64, (1, 1), mode='regular'),
            BottleneckResBlock(64, 64, (1, 1), mode='regular'),
            BottleneckResBlock(64, 64, (1, 1), mode='regular')
        )

        self.layer2 = [BottleneckResBlock(64, 128, (1, 1), mode='downsample')]
        for i in range(num_blocks[0] - 1):
            self.layer2.append(BottleneckResBlock(128, 128, (1, 1), mode='regular'))
        self.layer2 = nn.ModuleList(self.layer2)

        self.layer3 = [BottleneckResBlock(128, 256, (2, 2), mode='downsample')]
        for i in range(num_blocks[1] - 1):
            self.layer3.append(ResBlock(256, 256, (1, 1), mode='regular'))
        self.layer3 = nn.ModuleList(self.layer3)

        self.layer4 = [ResBlock(256, 512, (2, 2), mode='downsample')]
        for i in range(num_blocks[2] - 1):
            self.layer4.append(ResBlock(512, 512, (1, 1), mode='regular'))
        self.layer4 = nn.ModuleList(self.layer4)

        self.avgpool = nn.AvgPool2d(kernel_size = (8, 8))
        self.fc = nn.Linear(512, num_classes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        for i in range(len(self.layer2)):
            out = self.layer2[i](out)
        for i in range(len(self.layer3)):
            out = self.layer3[i](out)
        for i in range(len(self.layer4)):
            out = self.layer4[i](out)
        out = self.avgpool(out)
        out = nn.Flatten()(out)
        out = self.fc(out)
        return out