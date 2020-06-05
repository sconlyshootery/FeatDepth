#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yukun(kunyu@deepmotion.ai)


from __future__ import absolute_import, division, print_function

import torch.nn as nn

from .depth_encoder import DepthEncoder
from .depth_decoder import DepthDecoder


class DepthModel(nn.Module):

    def __init__(self, layer_num=50):

        super(DepthModel, self).__init__()

        self.DepthEncoder = DepthEncoder(layer_num, None)
        self.DepthDecoder = DepthDecoder(self.DepthEncoder.num_ch_enc)

    def forward(self, input):

        outputs = self.DepthDecoder(self.DepthEncoder(input))

        return outputs[("disp", 0, 0)]
