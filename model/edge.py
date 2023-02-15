import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


def get_sobel(in_chan, out_chan):
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)

    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))

    return sobel_x, sobel_y


def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input


class Downsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(y, 2, stride=2)

        return x, y


class ESAM(nn.Module):
    def __init__(self, in_channels):
        super(ESAM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.ban = nn.BatchNorm2d(1)
        self.sobel_x1, self.sobel_y1 = get_sobel(in_channels, 1)

    def forward(self, x):
        y = run_sobel(self.sobel_x1, self.sobel_y1, x)
        y = F.relu(self.bn(y))
        y = self.conv1(y)
        y = x + y
        y = self.conv2(y)
        y = F.relu(self.ban(y))

        return y


class Edgenet(nn.Module):
    def __init__(self):
        super(Edgenet, self).__init__()
        in_chan = 2
        self.down1 = Downsample_block(in_chan, 64)
        self.down2 = Downsample_block(64, 128)
        self.down3 = Downsample_block(128, 256)
        self.down4 = Downsample_block(256, 512)
        self.conv1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.up = ESAM(1)
        self.up1 = ESAM(128)
        self.up2 = ESAM(256)
        self.up3 = ESAM(512)
        self.up4 = ESAM(1024)

    def forward(self, x, feature=False):
        x, y1 = self.down1(x)   # 64
        x, y2 = self.down2(x)   # 128
        x, y3 = self.down3(x)   # 256
        x, y4 = self.down4(x)   # 512
        y5 = F.relu(self.bn1(self.conv1(x)))    # 1024
        out1 = self.up1(y2)

        out2 = self.up2(y3)
        out2 = F.interpolate(out2, scale_factor=2, mode='bilinear', align_corners=True)

        out3 = self.up3(y4)
        out3 = F.interpolate(out3, scale_factor=4, mode='bilinear', align_corners=True)

        out4 = self.up4(y5)
        out4 = F.interpolate(out4, scale_factor=8, mode='bilinear', align_corners=True)

        out = out1 + out2
        out = self.up(out)

        out = out + out3
        out = self.up(out)

        out = out + out4
        out = self.up(out)

        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)

        if feature:
            return y2, y3, y4, y5
        else:
            return out
