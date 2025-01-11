#!/usr/bin/env python

import torch
from torch import nn
import todos
import pdb

class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=(ksize - 1) // 2, # same padding
            groups=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        # assert shortcut == True
        # hidden_channels = out_channels
        self.conv1 = BaseConv(in_channels, out_channels, 1, stride=1)
        self.conv2 = BaseConv(out_channels, out_channels, 3, stride=1)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13)):
        super().__init__()
        # in_channels = 1024
        # out_channels = 1024
        # kernel_sizes = (5, 9, 13)

        hidden_channels = in_channels // 2 # 512
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        # (Pdb) pp self.m
        # ModuleList(
        #   (0): MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
        #   (1): MaxPool2d(kernel_size=9, stride=1, padding=4, dilation=1, ceil_mode=False)
        #   (2): MaxPool2d(kernel_size=13, stride=1, padding=6, dilation=1, ceil_mode=False)
        # )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1) # 2048
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""
    def __init__(self, in_channels, out_channels, n=1, shortcut=True):
        super().__init__()
        # assert shortcut == False
        hidden_channels = out_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1)
        module_list = [
            Bottleneck(hidden_channels, hidden_channels, shortcut) for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x1 = self.m(x1)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv3(x)
        return x


class Focus(nn.Module):
    """Focus width and height information into channel space."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize=3, stride=1)

    def forward(self, x):
        # tensor [x1] size: [1, 3, 640, 640], min: 0.0, max: 255.0, mean: 128.074554

        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]

        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        # tensor [patch_top_left] size: [1, 3, 320, 320], min: 0.0, max: 255.0, mean: 128.029221
        # tensor [patch_top_right] size: [1, 3, 320, 320], min: 0.0, max: 255.0, mean: 128.043732
        # tensor [patch_bot_left] size: [1, 3, 320, 320], min: 0.0, max: 255.0, mean: 128.126816
        # tensor [patch_bot_right] size: [1, 3, 320, 320], min: 0.0, max: 255.0, mean: 128.09848

        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        # tensor [x2] size: [1, 12, 320, 320], min: 0.0, max: 255.0, mean: 128.074554

        return self.conv(x)


class CSPDarknet(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_features = ("dark3", "dark4", "dark5")
        base_channels = 64
        base_depth = 3

        # stem
        self.stem = Focus(3, base_channels)

        # dark2
        self.dark2 = nn.Sequential(
            BaseConv(base_channels, base_channels * 2, 3, 2),
            CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, shortcut=True), # xxxx_debug
        )

        # dark3
        self.dark3 = nn.Sequential(
            BaseConv(base_channels * 2, base_channels * 4, 3, 2),
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, shortcut=True), # xxxx_debug
        )

        # dark4
        self.dark4 = nn.Sequential(
            BaseConv(base_channels * 4, base_channels * 8, 3, 2),
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, shortcut=True), # xxxx_debug
        )

        # dark5
        self.dark5 = nn.Sequential(
            BaseConv(base_channels * 8, base_channels * 16, 3, 2),
            SPPBottleneck(base_channels * 16, base_channels * 16),
            CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False),
        )

    def forward(self, x):
        # outputs = {}
        x = self.stem(x)
        # outputs["stem"] = x
        # d1 = x
        x = self.dark2(x)
        # outputs["dark2"] = x
        # d2 = x
        x = self.dark3(x)
        d3 = x
        # outputs["dark3"] = x
        x = self.dark4(x)
        # outputs["dark4"] = x
        d4 = x
        x = self.dark5(x)
        # outputs["dark5"] = x
        d5 = x
        # return {k: v for k, v in outputs.items() if k in self.out_features}
        # (d3, d4, d5) is tuple: len = 3
        #     tensor [item] size: [1, 256, 80, 80], min: -0.278465, max: 9.145242, mean: -0.011328
        #     tensor [item] size: [1, 512, 40, 40], min: -0.278465, max: 10.702929, mean: 0.043139
        #     tensor [item] size: [1, 1024, 20, 20], min: -0.278465, max: 8.779962, mean: 0.130543

        return d3, d4, d5
