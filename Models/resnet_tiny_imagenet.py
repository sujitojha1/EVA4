'''ResNet Tiny Imagenet in PyTorch.

Modified reset architecture for Tiny Imagenet

Reference:
[1] Code is copied from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        # self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
        #         nn.MaxPool2d(2, 2),
        #         nn.BatchNorm2d(planes),
        # )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        #self.in_planes = 64

        # PrepLayer 
        self.preplayer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Layer 1
        self.layer1_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer1_ResBlock = ResBlock(128, 128)

        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Layer 3
        self.layer3_conv = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer3_ResBlock = ResBlock(512, 512)

        self.layer3 = nn.Sequential()

        self.pool1 = nn.adaptive_avg_pool2d(1)

        # Fully Connected Layer
        self.fc = nn.Conv2d(512, 200, kernel_size=1, padding=0, bias=False)


    def forward(self, x):
        prep   = self.preplayer(x)
        x1     = self.layer1_conv(prep)
        R1     = self.layer1_ResBlock(x1)
        layer1 = x1 + R1
        layer2 = self.layer2(layer1)
        x3     = self.layer3_conv(layer2)
        R3     = self.layer3_ResBlock(x3)
        layer3 = self.layer3(x3 + R3)
        pool1  = self.pool1(layer3)
        fc     = self.fc(pool1)
        out    = fc.view(-1,200)
        return F.log_softmax(out, dim=-1)


def ResNet_tiny_imagenet():
    return ResNet()
