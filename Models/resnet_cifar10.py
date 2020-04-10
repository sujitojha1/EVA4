'''ResNet Cifar in PyTorch.

Modified reset architecture for CIFAR10

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
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(planes),
        )

    def forward(self, x):
        out  = F.relu(self.bn1(self.conv1(x)))
        out  = F.relu(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
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
        self.layer1 = ResBlock(64, 128)

        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Layer 3
        self.layer3 = ResBlock(256, 512)

        self.pool1 = nn.MaxPool2d(4,4)

        # Fully Connected Layer
        self.fc = nn.Conv2d(512, 10, kernel_size=1, padding=0, bias=False)


    def forward(self, x):
        out = self.preplayer(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool1(out)
        out = self.fc(out)
        out = out.view(-1,10)
        return F.log_softmax(out, dim=-1)


def ResNet_Cifar10():
    return ResNet()
