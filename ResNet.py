from typing import List
import torch.nn as nn

class BaseResidualBlock(nn.Module):
    """
    An inner block for a ResNet model.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module = None, **kwargs):
        super(BaseResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.downsample = downsample

        self.relu = nn.ReLU()

    def _block_forward(self, x):
        return x

    def forward(self, x):
        residual = x
        out = self._block_forward(x)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        return self.relu(out)

class ResidualBlock(BaseResidualBlock):
    """
    A basic block for a ResNet model.

    Consists of two Convolutional layers, transforming the data from N channels to M channels.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module = None, **kwargs):
        super(ResidualBlock, self).__init__(in_channels, out_channels, stride, downsample, **kwargs)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(out_channels))

    def _block_forward(self, x):
        out = self.conv1(x)
        return self.conv2(out)

class ResNetLayer:
    def __init__(self, block: BaseResidualBlock, num_blocks: int, planes: int, stride: int = 1, **kwargs):
        self.block = block
        self.num_blocks = num_blocks
        self.planes = planes
        self.stride = stride
        self.kwargs = kwargs

    def build(self, in_planes: int) -> nn.Module:
        downsample = None
        if self.stride != 1 or in_planes != self.planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.planes, kernel_size = 1, stride = self.stride),
                nn.BatchNorm2d(self.planes))
        blocks = [self.block(in_channels = in_planes, out_channels = self.planes, stride = self.stride, downsample = downsample, kwargs = self.kwargs)]

        for i in range(1, self.num_blocks):
            blocks.append(self.block(self.planes, self.planes))

        return nn.Sequential(*blocks)

class ResNet(nn.Module):
    def __init__(self, in_channels : int, num_classes: int, layers: List[ResNetLayer], preprocess: nn.Module = None, postprocess: nn.Module = None):
        super(ResNet, self).__init__()

        if len(layers) == 0:
            raise Exception("ResNet must have at least 1 layer")

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, layers[0].planes, kernel_size = 7, stride = 2, padding = 3),
            nn.BatchNorm2d(layers[0].planes),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        nn_layers = [layers[0].build(layers[0].planes)]
        for i in range(1, len(layers)):
            nn_layers.append(layers[i].build(layers[i - 1].planes))
        self.blocks = nn.Sequential(*nn_layers)

        self.avgpool = nn.AvgPool2d(kernel_size = 7, stride = 1)
        self.classify = nn.Linear(layers[len(layers) - 1].planes, num_classes)

        self.preprocess = preprocess
        self.postprocess = postprocess

    def forward(self, x):
        if self.preprocess:
            x = self.preprocess(x)

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.blocks(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)

        if self.postprocess:
            x = self.postprocess(x)

        return x
