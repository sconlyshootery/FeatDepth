from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
from .resnet import ResNet, BasicBlock, resnet18, resnet34, resnet50, resnet101, Bottleneck
from torch.nn import BatchNorm2d as bn


class ResNetMultiImageInput(ResNet):
    def __init__(self, block, layers, num_classes=1000, num_input_images=2):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = bn(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, num_input_images=2, pretrained_path=None):
    assert num_layers in [18, 34, 50, 101], "Can only run with 18, 34, 50, 101 layers resnet"
    blocks = {18 : [2, 2, 2,  2],
              34 : [3, 4, 6,  3],
              50 : [3, 4, 6,  3],
              101: [3, 4, 23, 3],
              }[num_layers]

    if num_layers < 40:
        model = ResNetMultiImageInput(BasicBlock, blocks, num_input_images=num_input_images)
    elif num_layers > 40:
        model = ResNetMultiImageInput(Bottleneck, blocks, num_input_images=num_input_images)

    if pretrained_path is not None:
        loaded = torch.load(pretrained_path)
        loaded['conv1.weight'] = torch.cat([loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class PoseEncoder(nn.Module):
    def __init__(self, num_layers, pretrained_path=None, num_input_images=2):
        super(PoseEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: resnet18,
                   34: resnet34,
                   50: resnet50,
                   101: resnet101,}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, num_input_images, pretrained_path)
        else:
            self.encoder = resnets[num_layers]()
            if pretrained_path is not None:
                checkpoint = torch.load(pretrained_path)
                self.encoder.load_state_dict(checkpoint)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        # for name, param in self.encoder.named_parameters():
        #     if 'bn' in name:
        #         param.requires_grad = False

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features
