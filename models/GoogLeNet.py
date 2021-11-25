import torch, torchvision
import os, sys
import torch.nn as nn
from torchsummary import summary

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding = 0):
        super(ConvBNReLU, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)

class Inception(nn.Module):
    def __init__(self, in_ch, b1_conv1, b2_conv1, b2_conv3, b3_conv1, b3_conv5, b4_conv):
        super(Inception, self).__init__()
        self.branch_1 = ConvBNReLU(in_ch, b1_conv1, kernel_size=(1, 1))

        self.branch_2 = nn.Sequential(
            ConvBNReLU(in_ch, b2_conv1, kernel_size=(1, 1)),
            ConvBNReLU(b2_conv1, b2_conv3, kernel_size=(3, 3), padding=1)
        )

        self.branch_3 = nn.Sequential(
            ConvBNReLU(in_ch, b3_conv1, kernel_size=(1, 1)),
            ConvBNReLU(b3_conv1, b3_conv5, kernel_size=(5, 5), padding=2)
        )

        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            ConvBNReLU(in_ch, b4_conv, kernel_size=(1, 1))
        )

    def forward(self, x):
        return torch.cat([
            self.branch_1(x),
            self.branch_2(x),
            self.branch_3(x),
            self.branch_4(x)
        ], dim=1)

class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.name = "GoogLeNet"
        self.pre_inception = nn.Sequential(
            ConvBNReLU(3, 64, kernel_size=(7, 7), padding=3),
            ConvBNReLU(64, 64, kernel_size=(1, 1)),
            ConvBNReLU(64, 192, kernel_size=(1, 1))
        )
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(1024, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.pre_inception(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)
        return x