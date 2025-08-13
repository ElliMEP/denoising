# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 14:40:59 2022

@author: e.pfaehler
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from PixelShuffle3D import PixelShuffle3d



class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv3d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv3d(channels, channels , kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w, d = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w * d)
        k = k.reshape(b, self.num_heads, -1, h * w * d)
        v = v.reshape(b, self.num_heads, -1, h * w * d)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w, d))
        return out


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor, up):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv3d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv3d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        if up ==True:
          self.project_out = nn.Conv3d(hidden_channels, channels * 2, kernel_size=1, bias=False)
        else:
          self.project_out = nn.Conv3d(hidden_channels, channels, kernel_size=1, bias=False)
    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor, up):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor, up)
        self.up = up
        print(channels)
    def forward(self, x):
        b, c, h, w, d = x.shape

        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w, d))
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                         .contiguous().reshape(b, c, h, w, d))
        return x

class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(nn.Conv3d(channels, channels * 8, kernel_size=3, padding=1, bias=False),
                                  PixelShuffle3d(2))

    def forward(self, x):

        return self.body(x)

class LastUpSample(nn.Module):
    def __init__(self, channels):
        super(LastUpSample, self).__init__()
        self.body = nn.Sequential(nn.Conv3d(channels, channels * 8, kernel_size=3, padding=1, bias=False),
                                  PixelShuffle3d(2))

    def forward(self, x):

        return self.body(x)
class SRFormer3D(nn.Module):
    def __init__(self, num_blocks=[4, 6, 6, 8], num_heads=[1, 2, 4, 8], channels=[8, 16, 32, 64], num_refinement=1,
                 expansion_factor=2.66):
        super(SRFormer3D, self).__init__()
        self.channels = channels
        self.embed_conv = nn.Conv3d(1, channels[0], kernel_size=3, padding=1, bias=False)
        self.conv1 = nn.Conv3d(channels[0], channels[1],kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv3d(channels[1], channels[2],kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv3d(channels[2], channels[1],kernel_size=3, padding=1, bias=False)
        self.encoder1 = nn.Sequential(*[TransformerBlock(channels[0], num_heads[1], expansion_factor, False)] )

        self.decoder1 = nn.Sequential(*[TransformerBlock(channels[2], num_heads[3], expansion_factor, False)])
        self.decoder2 = nn.Sequential(*[TransformerBlock(channels[3], num_heads[3], expansion_factor, False)])
        self.UpSample = UpSample(channels[2])
        self.LastUp = LastUpSample(channels[3])
        self.decoder3 = nn.Sequential(*[TransformerBlock(channels[2], num_heads[3], expansion_factor, False)])
        self.refinement = nn.Sequential(*[TransformerBlock(channels[0] , num_heads[0], expansion_factor, False)
                                          for _ in range(num_refinement)])
        self.conv4 = nn.Conv3d(channels[1], channels[0], kernel_size=1, padding='same', bias=False)
        # self.conv5 = nn.Conv2d(channels[2], channels[1], kernel_size=1, padding='same', bias=False)
        self.output = nn.Conv3d(channels[1], 1, kernel_size=1, padding='same', bias=False)
        self.doprint = False
        self.upConv = nn.Conv3d(1,channels[0], kernel_size=1, padding='same', bias=False)

    def forward(self, x):
        fo = self.embed_conv(x)
        out_enc1 = self.encoder1(fo)
        out_conv1 = self.conv1(out_enc1)
        out_conv2 = self.conv2(out_conv1)

        out_dec0 = self.decoder1(out_conv2)

        out_conv3 = self.conv3(out_dec0)
        out_conv4 = self.conv4(out_conv3)

        fr = self.refinement(out_conv4)

        out = self.output(torch.cat([fr,self.upConv(x)], dim=1))
        if self.doprint:
            print("\tIn Model: input size", x.size(),
              "output size", out.size())
            self.doprint=False
        return out
