import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import BACKBONES
from .resnet_tsm import ResNetTSM


@BACKBONES.register_module()
class ResNetTDN(ResNetTSM):
    """TDN."""

    def __init__(self, depth, num_segments=8, **kwargs):
        super().__init__(depth, **kwargs)
        # TODO Conversion logic
        if num_segments == 8:
            self.apha = 0.5
            self.belta = 0.5
        else:
            self.apha = 0.75
            self.belta = 0.25
        # TODO Checkout what they are doing
        self.input_size = 224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]

        self.conv1_5 = nn.Sequential(
            nn.Conv2d(
                4 * self.in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.maxpool_diff = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.avg_diff = nn.AvgPool2d(kernel_size=2, stride=2)

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        super().init_weights()
        # implement conv1_5 and inflate weight
        params = [
            x.clone().detach().requires_grad_(True)
            for x in self.conv1.conv.parameters()
        ]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (
            4 * self.in_channels, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(
            dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        self.conv1_5[0].weight.data = new_kernels
        self.resnext_layer1 = copy.deepcopy(getattr(self, self.res_layers[0]))
        # Or _load_torchvision_checkpoint() in ResNet will automatically
        # trying to assign it parameters in ckpt

    def forward(self, x):
        # [N, C, 5, H, W]
        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]
        x3 = x[:, :, 2, :, :]
        x4 = x[:, :, 3, :, :]
        x5 = x[:, :, 4, :, :]
        # [N, C, H, W]
        x_c5 = self.conv1_5(
            self.avg_diff(
                torch.cat([x2 - x1, x3 - x2, x4 - x3, x5 - x4],
                          1).view((-1, 4 * self.in_channels) + x.size()[-2:])))
        x_diff = self.maxpool_diff(1.0 / 1.0 * x_c5)

        temp_out_diff1 = x_diff
        x_diff = self.resnext_layer1(x_diff)

        x = self.conv1(x3)
        # fusion layer1
        x = self.maxpool(x)
        temp_out_diff1 = F.interpolate(temp_out_diff1, x.size()[2:])
        x = self.apha * x + self.belta * temp_out_diff1
        # fusion layer2
        x = self.layer1(x)
        x_diff = F.interpolate(x_diff, x.size()[2:])
        x = self.apha * x + self.belta * x_diff

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # [N, 2048, 7, 7]

        return x
