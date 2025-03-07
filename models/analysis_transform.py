#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.autograd
import sys
import os
sys.path.append('/media/zll/d/Repo/gsb/code/FDEGCNet/models')#Absolute path to "models"
from GDN import GDN
from ODAM import ODAM


class Analysis_transform(nn.Module):
    def __init__(self, num_filters=128):
        super(Analysis_transform, self).__init__()
        # i = 0
        self.b0_shortcut = nn.Conv2d(3, num_filters, 1, stride=2)
        self.b0_layer2 = nn.Conv2d(3, num_filters, 3, stride=2, padding=1)
        self.b0_layer2_relu = nn.LeakyReLU()
        self.b0_layer3 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b0_layer3_GDN = GDN(num_filters)

        # i = 1
        self.b1_layer0 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b1_layer0_relu = nn.LeakyReLU()
        self.b1_layer1 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b1_layer1_relu = nn.LeakyReLU()
        self.b1_shortcut = nn.Conv2d(num_filters, num_filters, 1, stride=2)
        self.b1_layer2 = nn.Conv2d(num_filters, num_filters, 3, stride=2, padding=1)
        self.b1_layer2_relu = nn.LeakyReLU()
        self.b1_layer3 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b1_layer3_GDN = GDN(num_filters)
        self.attention1 = ODAM(num_filters)

        # i = 2
        self.b2_layer0 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b2_layer0_relu = nn.LeakyReLU()
        self.b2_layer1 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b2_layer1_relu = nn.LeakyReLU()
        self.b2_shortcut = nn.Conv2d(num_filters, num_filters, 1, stride=2)
        self.b2_layer2 = nn.Conv2d(num_filters, num_filters, 3, stride=2, padding=1)
        self.b2_layer2_relu = nn.LeakyReLU()
        self.b2_layer3 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b2_layer3_GDN = GDN(num_filters)

        # i = 3
        self.b3_layer0 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b3_layer0_relu = nn.LeakyReLU()
        self.b3_layer1 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b3_layer1_relu = nn.LeakyReLU()
        self.b3_layer2 = nn.Conv2d(num_filters, num_filters, 3, stride=2, padding=1, bias=False)
        self.attention2 = ODAM(num_filters)


    def forward(self, x):
        # i = 0
        shortcut0 = self.b0_shortcut(x)# 下采样
        b0 = self.b0_layer2(x)# 卷积+下采样
        b0 = self.b0_layer2_relu(b0)# 激活
        b0 = self.b0_layer3(b0) # 卷积
        b0 = self.b0_layer3_GDN(b0)# 归一化
        b0 += shortcut0

        # i = 1
        b1 = self.b1_layer0(b0)
        b1 = self.b1_layer0_relu(b1)
        b1 = self.b1_layer1(b1)
        b1 = self.b1_layer1_relu(b1)
        b1 += b0
        shortcut1 = self.b1_shortcut(b1)
        b1 = self.b1_layer2(b1)
        b1 = self.b1_layer2_relu(b1)
        b1 = self.b1_layer3(b1)
        b1 = self.b1_layer3_GDN(b1)
        b1 += shortcut1
        b1 = self.attention1(b1)

        # i = 2
        b2 = self.b2_layer0(b1)
        b2 = self.b2_layer0_relu(b2)
        b2 = self.b2_layer1(b2)
        b2 = self.b2_layer1_relu(b2)
        b2 += b1
        shortcut2 = self.b2_shortcut(b2)
        b2 = self.b2_layer2(b2)
        b2 = self.b2_layer2_relu(b2)
        b2 = self.b2_layer3(b2)
        b2 = self.b2_layer3_GDN(b2)
        b2 += shortcut2

        # i = 3
        b3 = self.b3_layer0(b2)
        b3 = self.b3_layer0_relu(b3)
        b3 = self.b3_layer1(b3)
        b3 = self.b3_layer1_relu(b3)
        b3 += b2
        b3 = self.b3_layer2(b3)
        b3 = self.attention2(b3)

        return b3

if __name__ == "__main__":
    Analysis_transform = Analysis_transform()
    input_image = torch.zeros([1,3,256,256])
    feature = Analysis_transform(input_image)
    print(feature.shape)
