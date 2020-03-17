'''Quiz S9, DNN

DNN Structure
x1 = Input
x2 = Conv(x1)
x3 = Conv(x1 + x2)
x4 = MaxPooling(x1 + x2 + x3)
x5 = Conv(x4)
x6 = Conv(x4 + x5)
x7 = Conv(x4 + x5 + x6)
x8 = MaxPooling(x5 + x6 + x7)
x9 = Conv(x8)
x10 = Conv (x8 + x9)
x11 = Conv (x8 + x9 + x10)
x12 = GAP(x11)
x13 = FC(x12)

'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()

        # Input layer - x1
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        
        # x2 = conv(x1)
        self.b2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        # x3 = conv(x1 + x2)
        self.b3 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        # x4 = maxpool(x1 + x2 + x3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # x5 = conv(x4)
        self.b4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

        # x6 = conv(x4 + x5)
        self.b5 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

        # x7 = conv(x4 + x5 + x6)
        self.b6 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        # x9 = conv(x8)
        self.b7 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        # x10 = conv(x8 + x9)
        self.b8 = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        # x11 = conv(x8 + x9 + x10)
        self.b9 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
        )

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, 10)

    def forward(self, x):
        x1 = self.b1(x)
        x2 = self.b2(x1)
        x3 = self.b3(torch.cat([x1,x2], 1))
        x4 = self.maxpool(torch.cat([x1,x2,x3], 1))
        x5 = self.b4(x4)
        x6 = self.b5(torch.cat([x4,x5], 1))
        x7 = self.b6(torch.cat([x4,x5,x6], 1))
        x8 = self.maxpool(torch.cat([x5,x6,x7], 1))
        x9 = self.b7(x8)
        x10 = self.b8(torch.cat([x8,x9], 1))
        x11 = self.b9(torch.cat([x8,x9,x10], 1))
        x12 = self.avgpool(x11)
        x13 = x12.view(x12.size(0), -1)
        out = self.linear(x13)
        return out