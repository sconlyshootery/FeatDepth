import torch
import torch.nn as nn
from .layers import Conv3x3, upsample, ConvBlock


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_output_channels=1):
        super(DepthDecoder, self).__init__()

        num_ch_dec = [16, 32, 64, 128, 256]

        # upconv
        self.upconv5 = ConvBlock(num_ch_enc[4], num_ch_dec[4])
        self.upconv4 = ConvBlock(num_ch_dec[4], num_ch_dec[3])
        self.upconv3 = ConvBlock(num_ch_dec[3], num_ch_dec[2])
        self.upconv2 = ConvBlock(num_ch_dec[2], num_ch_dec[1])
        self.upconv1 = ConvBlock(num_ch_dec[1], num_ch_dec[0])

        # iconv
        self.iconv5 = ConvBlock(num_ch_dec[4] + num_ch_enc[3], num_ch_dec[4])
        self.iconv4 = ConvBlock(num_ch_dec[3] + num_ch_enc[2], num_ch_dec[3])
        self.iconv3 = ConvBlock(num_ch_dec[2] + num_ch_enc[1], num_ch_dec[2])
        self.iconv2 = ConvBlock(num_ch_dec[1] + num_ch_enc[0], num_ch_dec[1])
        self.iconv1 = ConvBlock(num_ch_dec[0]                , num_ch_dec[0])

        # disp
        self.disp4 = Conv3x3(num_ch_dec[3], num_output_channels)
        self.disp3 = Conv3x3(num_ch_dec[2], num_output_channels)
        self.disp2 = Conv3x3(num_ch_dec[1], num_output_channels)
        self.disp1 = Conv3x3(num_ch_dec[0], num_output_channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features, frame_id=0):
        self.outputs = {}
        econv1, econv2, econv3, econv4, econv5 = input_features
        # (64,64,128,256,512)*4

        upconv5 = upsample(self.upconv5(econv5))
        iconv5 = self.iconv5(torch.cat((upconv5, econv4), 1))

        upconv4 = upsample(self.upconv4(iconv5))
        iconv4 = self.iconv4(torch.cat((upconv4, econv3), 1))

        upconv3 = upsample(self.upconv3(iconv4))
        iconv3 = self.iconv3(torch.cat((upconv3, econv2), 1))

        upconv2 = upsample(self.upconv2(iconv3))
        iconv2 = self.iconv2(torch.cat((upconv2, econv1), 1))

        upconv1 = upsample(self.upconv1(iconv2))
        iconv1 = self.iconv1(upconv1)

        self.outputs[("disp", frame_id, 3)] = self.sigmoid(self.disp4(iconv4))
        self.outputs[("disp", frame_id, 2)] = self.sigmoid(self.disp3(iconv3))
        self.outputs[("disp", frame_id, 1)] = self.sigmoid(self.disp2(iconv2))
        self.outputs[("disp", frame_id, 0)] = self.sigmoid(self.disp1(iconv1))
        return self.outputs
