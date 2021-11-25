import torch, torchvision
import os, sys
import torch.nn as nn
from torchsummary import summary

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(

        )