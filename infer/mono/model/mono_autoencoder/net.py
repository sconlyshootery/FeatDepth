from __future__ import absolute_import, division, print_function
import torch
import torch.nn.functional as F
import torch.nn as nn

import os
import matplotlib.pyplot as plt

from .layers import SSIM
from .encoder import Encoder
from .decoder import Decoder
from ..registry import MONO


@MONO.register_module
class autoencoder(nn.Module):
    def __init__(self, options):
        super(autoencoder, self).__init__()
        self.opt = options

        self.Encoder = Encoder(self.opt.depth_num_layers, self.opt.depth_pretrained_path)
        self.Decoder = Decoder(self.Encoder.num_ch_enc)

        self.ssim = SSIM()
        self.count = 0

    def forward(self, inputs):
        features = self.Encoder(inputs[("color", 0, 0)])
        outputs = self.Decoder(features, 0)
        if self.training:
            loss_dict = self.compute_losses(inputs, outputs, features)
            return outputs, loss_dict
        return outputs

    def robust_l1(self, pred, target):
        eps = 1e-3
        return torch.sqrt(torch.pow(target - pred, 2) + eps ** 2)

    def compute_reprojection_loss(self, pred, target):
        photometric_loss = self.robust_l1(pred, target).mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = (0.85 * ssim_loss + 0.15 * photometric_loss)
        return reprojection_loss

    def compute_losses(self, inputs, outputs, features):
        loss_dict = {}
        interval = 1000
        target = inputs[("color", 0, 0)]
        for i in range(5):
            f=features[i]
            smooth_loss = self.get_smooth_loss(f, target)
            loss_dict[('smooth_loss', i)] = smooth_loss/ (2 ** i)/5

        for scale in self.opt.scales:
            """
            initialization
            """
            pred = outputs[("disp", 0, scale)]

            _,_,h,w = pred.size()
            target = F.interpolate(target, [h, w], mode="bilinear", align_corners=False)
            min_reconstruct_loss = self.compute_reprojection_loss(pred, target)
            loss_dict[('min_reconstruct_loss', scale)] = min_reconstruct_loss.mean()/len(self.opt.scales)

            if self.count % interval == 0:
                img_path = os.path.join('/node01_data5/monodepth2-test/odo', 'auto_{:0>4d}_{}.png'.format(self.count // interval, scale))
                plt.imsave(img_path, pred[0].transpose(0,1).transpose(1,2).data.cpu().numpy())
                img_path = os.path.join('/node01_data5/monodepth2-test/odo', 'img_{:0>4d}_{}.png'.format(self.count // interval, scale))
                plt.imsave(img_path, target[0].transpose(0, 1).transpose(1, 2).data.cpu().numpy())

        self.count += 1
        return loss_dict

    def get_smooth_loss(self, disp, img):
        b, _, h, w = disp.size()
        img = F.interpolate(img, (h, w), mode='area')

        disp_dx, disp_dy = self.gradient(disp)
        img_dx, img_dy = self.gradient(img)

        disp_dxx, disp_dxy = self.gradient(disp_dx)
        disp_dyx, disp_dyy = self.gradient(disp_dy)

        img_dxx, img_dxy = self.gradient(img_dx)
        img_dyx, img_dyy = self.gradient(img_dy)

        smooth1 = torch.mean(disp_dx.abs() * torch.exp(-img_dx.abs().mean(1, True))) + \
                  torch.mean(disp_dy.abs() * torch.exp(-img_dy.abs().mean(1, True)))

        smooth2 = torch.mean(disp_dxx.abs() * torch.exp(-img_dxx.abs().mean(1, True))) + \
                  torch.mean(disp_dxy.abs() * torch.exp(-img_dxy.abs().mean(1, True))) + \
                  torch.mean(disp_dyx.abs() * torch.exp(-img_dyx.abs().mean(1, True))) + \
                  torch.mean(disp_dyy.abs() * torch.exp(-img_dyy.abs().mean(1, True)))

        return -self.opt.dis * smooth1+ self.opt.cvt * smooth2

    def gradient(self, D):
        dy = D[:, :, 1:] - D[:, :, :-1]
        dx = D[:, :, :, 1:] - D[:, :, :, :-1]
        return dx, dy

