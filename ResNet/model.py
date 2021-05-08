import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.act = activation

    def forward(self, x):
        return self.relu( self.bn( self.conv(x) ) ) if self.act else self.bn( self.conv(x) )
    

class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, identity_downsample=None, stride=1
    ):
        super().__init__()
        self.layers = nn.Sequential(
            CNNBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            CNNBlock(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            CNNBlock(out_channels, out_channels*4, activation=False, kernel_size=1, stride=1, padding=0),
        )
        self.identity_downsample = identity_downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        x = self.layers(x)

        if self.identity_downsample is not None:
            x += self.identity_downsample(identity)

        x = self.relu(x)

        return x

class ResNet(nn.Module):
    def __init__(self, block, num_repeats, image_channels, num_classes):
        super().__init__()
        self.in_channels = 64
        self.conv1 = CNNBlock(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, num_repeats[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, num_repeats[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, num_repeats[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, num_repeats[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x
        
    def _make_layer(self, block, num_repeat, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = CNNBlock(self.in_channels, out_channels*4, activation=False, kernel_size=1, stride=stride)

        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels*4

        for i in range(num_repeat - 1):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)