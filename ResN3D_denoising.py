# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 19:58:45 2024

@author: e.pfaehler
"""
import torch
import torch.nn as nn
import math

class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.BatchNorm3d(64, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity_data = x
        output = self.conv1(x)
  #      output = self.in1(output)
        output = self.relu(output)
        output = self.conv1(output)
  #      output = self.in1(output)
        output = torch.add(output,identity_data)
        return output


class Generator(nn.Module):
    def __init__(self, nrChannels):
        super(Generator, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv3d(in_channels=nrChannels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(1):
            layers.append(self.constructResidualBlocks(_Residual_Block,5))
            layers.append(nn.BatchNorm3d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm3d(32))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv3d(in_channels=32, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False))
        self.resn = nn.Sequential(*layers)
    def constructResidualBlocks(self, block, nrBlocks):
         layers = []
         for _ in range(nrBlocks):
              layers.append(block())
         return nn.Sequential(*layers)

    def forward(self, x):
        out = self.resn(x)
        return out
