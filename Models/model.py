import torch
import torch.nn as nn
import torch.nn.functional as F

class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()        
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        return self.leakyreluB( self.convB( self.leakyreluA(self.convA( x ) ) )  )

class DownSample(nn.Sequential):
    expansion = 1

    #ResBlock
    def __init__(self, in_planes, planes, stride=1):
        super(DownSample, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()       

        self.in_planes = 64

        ### Encoder block

        # Prep Layer - Two images 3 channels + 3 channels concat 

        # Dimension 224x224x6
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False) # Dimension 112x112x64
        self.bn1 = nn.BatchNorm2d(64)
        
        #self.maxpool = nn.MaxPool2d(3, stride=2,padding=1) 

        self.layer1 = self._make_layer(DownSample, 64, 2, stride=1)  # Dimension 112x112x64
        self.layer2 = self._make_layer(DownSample, 128, 2, stride=2) # Dimension 56x56x128
        self.layer3 = self._make_layer(DownSample, 256, 2, stride=2) # Dimension 28x28x256
        self.layer4 = self._make_layer(DownSample, 512, 2, stride=2) # Dimension 14x14x512


        #self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=1)
        ### Decoder block
        self.up1 = UpSample(skip_input=512, output_features=256)
        self.up2 = UpSample(skip_input=256, output_features=128)
        self.up3 = UpSample(skip_input=128, output_features=64)
        self.up4 = UpSample(skip_input=64, output_features=64)

        self.conv3 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, inputs):

        bg_inp    = inputs['bg']
        fg_bg_inp = inputs['fg_bg']

        x = torch.cat([bg_inp,fg_bg_inp],dim=1)

        # Prep Layer
        out = F.relu(self.bn1(self.conv1(x)))

        # Downsample layers - ResBlock
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out4 = F.interpolate(out4, scale_factor=2, mode='bilinear', align_corners=True)

        # Upsample for mask
        x_d1 = self.up1(out4) + out3
        x_d1 = F.interpolate(x_d1, scale_factor=2, mode='bilinear', align_corners=True)
        x_d2 = self.up2(x_d1) + out2
        x_d2 = F.interpolate(x_d2, scale_factor=2, mode='bilinear', align_corners=True)
        x_d3 = self.up3(x_d2) + out1
        x_d3 = F.interpolate(x_d3, scale_factor=2, mode='bilinear', align_corners=True)
        x_d4 = self.up4(x_d3) + out
        x_d4 = F.interpolate(x_d4, scale_factor=2, mode='bilinear', align_corners=True)


        # Upsample for depth
        y_d1 = self.up1(out4) + out3
        y_d1 = F.interpolate(y_d1, scale_factor=2, mode='bilinear', align_corners=True)
        y_d2 = self.up2(y_d1) + out2
        y_d2 = F.interpolate(y_d2, scale_factor=2, mode='bilinear', align_corners=True)
        y_d3 = self.up3(y_d2) + out1
        y_d3 = F.interpolate(y_d3, scale_factor=2, mode='bilinear', align_corners=True)
        y_d4 = self.up4(y_d3) + out
        y_d4 = F.interpolate(y_d4, scale_factor=2, mode='bilinear', align_corners=True)

        return self.conv3(x_d4),self.conv3(y_d4)


