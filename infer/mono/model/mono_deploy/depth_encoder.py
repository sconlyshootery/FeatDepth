from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
from .resnet import resnet18, resnet34, resnet50, resnet101


class DepthEncoder(nn.Module):
    def __init__(self, num_layers, pretrained_path=None):
        super(DepthEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: resnet18,
                   34: resnet34,
                   50: resnet50,
                   101: resnet101,}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))


        self.encoder = resnets[num_layers]()
        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path)
            self.encoder.load_state_dict(checkpoint, strict=False)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x1 = self.encoder.relu(x)
        x2 = self.encoder.layer1(self.encoder.maxpool(x1))
        x3 = self.encoder.layer2(x2)
        x4 = self.encoder.layer3(x3)
        x5 = self.encoder.layer4(x4)
        self.features.append(x1)
        self.features.append(x2)
        self.features.append(x3)
        self.features.append(x4)
        self.features.append(x5)

        return self.features
