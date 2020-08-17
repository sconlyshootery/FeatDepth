# from __future__ import absolute_import, division, print_function
# import torch.nn as nn
# from torch import cat
# from .layers import ConvBlock, Conv3x3, upsample
#
#
# class DepthDecoder(nn.Module):
#     def __init__(self, num_ch_enc, num_output_channels=1):
#         super(DepthDecoder, self).__init__()
#
#         num_ch_dec = [16, 32, 64, 128, 256]
#
#         # upconv
#         self.upconv5 = ConvBlock(num_ch_enc[4], num_ch_dec[4])
#         self.upconv4 = ConvBlock(num_ch_dec[4], num_ch_dec[3])
#         self.upconv3 = ConvBlock(num_ch_dec[3], num_ch_dec[2])
#         self.upconv2 = ConvBlock(num_ch_dec[2], num_ch_dec[1])
#         self.upconv1 = ConvBlock(num_ch_dec[1], num_ch_dec[0])
#
#         # iconv
#         self.iconv5 = ConvBlock(num_ch_dec[4] + num_ch_enc[3], num_ch_dec[4])
#         self.iconv4 = ConvBlock(num_ch_dec[3] + num_ch_enc[2], num_ch_dec[3])
#         self.iconv3 = ConvBlock(num_ch_dec[2] + num_ch_enc[1], num_ch_dec[2])
#         self.iconv2 = ConvBlock(num_ch_dec[1] + num_ch_enc[0], num_ch_dec[1])
#         self.iconv1 = ConvBlock(num_ch_dec[0]                , num_ch_dec[0])
#
#         # disp
#         self.disp4 = Conv3x3(num_ch_dec[3], num_output_channels)
#         self.disp3 = Conv3x3(num_ch_dec[2], num_output_channels)
#         self.disp2 = Conv3x3(num_ch_dec[1], num_output_channels)
#         self.disp1 = Conv3x3(num_ch_dec[0], num_output_channels)
#
#         self.sigmoid = nn.Sigmoid()
#
#
#     def forward(self, input_features, frame_id=0):
#         self.outputs = {}
#         econv1, econv2, econv3, econv4, econv5 = input_features
#         # (64,64,128,256,512)*4
#
#         upconv5 = upsample(self.upconv5(econv5))
#         iconv5 = self.iconv5(cat((upconv5, econv4), 1))
#
#         upconv4 = upsample(self.upconv4(iconv5))
#         iconv4 = self.iconv4(cat((upconv4, econv3), 1))
#
#         upconv3 = upsample(self.upconv3(iconv4))
#         iconv3 = self.iconv3(cat((upconv3, econv2), 1))
#
#         upconv2 = upsample(self.upconv2(iconv3))
#         iconv2 = self.iconv2(cat((upconv2, econv1), 1))
#
#         upconv1 = upsample(self.upconv1(iconv2))
#         iconv1 = self.iconv1(upconv1)
#
#         self.outputs[("disp", frame_id, 3)] = self.sigmoid(self.disp4(iconv4))
#         self.outputs[("disp", frame_id, 2)] = self.sigmoid(self.disp3(iconv3))
#         self.outputs[("disp", frame_id, 1)] = self.sigmoid(self.disp2(iconv2))
#         self.outputs[("disp", frame_id, 0)] = self.sigmoid(self.disp1(iconv1))
#         return self.outputs


import torch
import torch.nn as nn
from .layers import Conv1x1, Conv3x3, CRPBlock, upsample


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc):
        super(DepthDecoder, self).__init__()

        bottleneck = 256
        stage = 4
        self.do = nn.Dropout(p=0.5)

        self.reduce4 = Conv1x1(num_ch_enc[4], 512, bias=False)
        self.reduce3 = Conv1x1(num_ch_enc[3], bottleneck, bias=False)
        self.reduce2 = Conv1x1(num_ch_enc[2], bottleneck, bias=False)
        self.reduce1 = Conv1x1(num_ch_enc[1], bottleneck, bias=False)

        self.iconv4 = Conv3x3(512, bottleneck)
        self.iconv3 = Conv3x3(bottleneck * 2 + 1, bottleneck)
        self.iconv2 = Conv3x3(bottleneck * 2 + 1, bottleneck)
        self.iconv1 = Conv3x3(bottleneck * 2 + 1, bottleneck)

        self.crp4 = self._make_crp(bottleneck, bottleneck, stage)
        self.crp3 = self._make_crp(bottleneck, bottleneck, stage)
        self.crp2 = self._make_crp(bottleneck, bottleneck, stage)
        self.crp1 = self._make_crp(bottleneck, bottleneck, stage)

        self.merge4 = Conv3x3(bottleneck, bottleneck)
        self.merge3 = Conv3x3(bottleneck, bottleneck)
        self.merge2 = Conv3x3(bottleneck, bottleneck)
        self.merge1 = Conv3x3(bottleneck, bottleneck)

        # disp
        self.disp4 = nn.Sequential(Conv3x3(bottleneck, 1), nn.Sigmoid())
        self.disp3 = nn.Sequential(Conv3x3(bottleneck, 1), nn.Sigmoid())
        self.disp2 = nn.Sequential(Conv3x3(bottleneck, 1), nn.Sigmoid())
        self.disp1 = nn.Sequential(Conv3x3(bottleneck, 1), nn.Sigmoid())

        self.nonlin = nn.ReLU(inplace=True)

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes, stages)]
        return nn.Sequential(*layers)

    def forward(self, input_features, frame_id=0):
        self.outputs = {}
        l0, l1, l2, l3, l4 = input_features

        x4 = self.reduce4(l4)
        x4 = self.iconv4(x4)
        x4 = self.nonlin(x4)
        x4 = self.crp4(x4)
        x4 = self.merge4(x4)
        x4 = self.nonlin(x4)
        x4 = upsample(x4)
        disp4 = self.disp4(x4)

        x3 = self.reduce3(l3)
        x3 = torch.cat((x3, x4, disp4), 1)
        x3 = self.iconv3(x3)
        x3 = self.nonlin(x3)
        x3 = self.crp3(x3)
        x3 = self.merge3(x3)
        x3 = self.nonlin(x3)
        x3 = upsample(x3)
        disp3 = self.disp3(x3)

        x2 = self.reduce2(l2)
        x2 = torch.cat((x2, x3, disp3), 1)
        x2 = self.iconv2(x2)
        x2 = self.nonlin(x2)
        x2 = self.crp2(x2)
        x2 = self.merge2(x2)
        x2 = self.nonlin(x2)
        x2 = upsample(x2)
        disp2 = self.disp2(x2)

        x1 = self.reduce1(l1)
        x1 = torch.cat((x1, x2, disp2), 1)
        x1 = self.iconv1(x1)
        x1 = self.nonlin(x1)
        x1 = self.crp1(x1)
        x1 = self.merge1(x1)
        x1 = self.nonlin(x1)
        x1 = upsample(x1)
        disp1 = self.disp1(x1)

        self.outputs[("disp", frame_id, 3)] = disp4
        self.outputs[("disp", frame_id, 2)] = disp3
        self.outputs[("disp", frame_id, 1)] = disp2
        self.outputs[("disp", frame_id, 0)] = disp1

        return self.outputs
