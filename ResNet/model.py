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