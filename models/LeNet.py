import torch
import torch.nn as nn

class LeNet(nn.Module) :
    def __init__(self) :
        super(LeNet, self).__init__()
        self.name = "LeNet"
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=(5, 5)),
            nn.ReLU()
        )
        self.pool_layer1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.ReLU()
        )
        self.pool_layer2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.C5_layer = nn.Sequential(
            nn.Linear(5*5*16, 120),
            nn.ReLU()
        )
        self.fc_layer1 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc_layer2 = nn.Linear(84, 10)

    def forward(self, x) :
        output = self.conv_layer1(x)
        output = self.pool_layer1(output)
        output = self.conv_layer2(output)
        output = self.pool_layer2(output)
        output = output.view(-1,5*5*16)
        output = self.C5_layer(output)
        output = self.fc_layer1(output)
        output = self.fc_layer2(output)
        return output
