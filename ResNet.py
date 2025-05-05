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

class BottleneckBlock(BaseResidualBlock):
    """
    A bottleneck Residual Block for use in a ResNet-50+ model.

    Consists of three convolutional players, transforming the data from N channels to M channels, while reducing
    the dimensionality of the data from the previous block.
    """
    def __init__(self, in_channels: int, out_channels: int, reduction_channels: int = 32, stride: int = 1, downsample: nn.Module = None, **kwargs):
        super(BottleneckBlock, self).__init__(in_channels, out_channels, stride, downsample, **kwargs)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, reduction_channels, kernel_size = 1, stride = stride),
            nn.BatchNorm2d(reduction_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(reduction_channels, reduction_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(reduction_channels),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(reduction_channels, out_channels, kernel_size = 1, stride = 1),
            nn.BatchNorm2d(out_channels))

    def _block_forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

class ResiduaLayer:
    def __init__(self, block: BaseResidualBlock, num_blocks: int, in_planes: int, out_planes: int, stride: int = 1, **kwargs):
        self.block = block
        self.num_blocks = num_blocks
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        self.kwargs = kwargs

    def build(self) -> nn.Module:
        downsample = None
        if self.stride != 1 or self.in_planes != self.out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, self.out_planes, kernel_size = 1, stride = self.stride),
                nn.BatchNorm2d(self.out_planes))
        blocks = [self.block(in_channels = self.in_planes, out_channels = self.out_planes, stride = self.stride, downsample = downsample, kwargs = self.kwargs)]

        for i in range(1, self.num_blocks):
            blocks.append(self.block(in_channels = self.out_planes, out_channels = self.out_planes))

        return nn.Sequential(*blocks)

class BottleneckLayer(ResiduaLayer):
    def __init__(self, num_blocks: int, in_planes: int, out_planes: int, reduction_planes: int, stride: int = 1, **kwargs):
        super(BottleneckLayer, self).__init__(BottleneckBlock, num_blocks, in_planes, out_planes, stride, **kwargs)
        self.reduction_planes = reduction_planes

    def build(self) -> nn.Module:
        downsample = None
        if self.stride != 1 or self.in_planes != self.out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, self.out_planes, kernel_size=1, stride=self.stride),
                nn.BatchNorm2d(self.out_planes))
        blocks = [BottleneckBlock(in_channels=self.in_planes, out_channels=self.out_planes, reduction_channels=self.reduction_planes, stride=self.stride, downsample=downsample, kwargs=self.kwargs)]

        for i in range(1, self.num_blocks):
            blocks.append(BottleneckBlock(in_channels = self.out_planes, out_channels = self.out_planes, reduction_channels=self.reduction_planes))

        return nn.Sequential(*blocks)

class ResNet(nn.Module):
    def __init__(self, in_channels : int, num_classes: int, layers: List[ResiduaLayer], preprocess: nn.Module = None, postprocess: nn.Module = None):
        super(ResNet, self).__init__()

        if len(layers) == 0:
            raise Exception("ResNet must have at least 1 layer")

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, layers[0].in_planes, kernel_size = 7, stride = 2, padding = 3),
            nn.BatchNorm2d(layers[0].in_planes),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        nn_layers = [layers[0].build()]
        for i in range(1, len(layers)):
            nn_layers.append(layers[i].build())
        self.blocks = nn.Sequential(*nn_layers)

        self.avgpool = nn.AvgPool2d(kernel_size = 7, stride = 1)
        self.classify = nn.Linear(layers[len(layers) - 1].out_planes, num_classes)

        self.preprocess = preprocess
        self.postprocess = postprocess

    def forward(self, x, *args, **kwargs):
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
