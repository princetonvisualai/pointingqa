import numpy as np
import torch.nn as nn
BatchNorm = nn.BatchNorm2d

#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=padding, bias=False, dilation=dilation)
# pylint: disable=too-many-instance-attributes, too-many-arguments
# pylint: disable=missing-docstring,unused-argument,arguments-differ


def conv_dil(kernel_size, in_planes, out_planes, stride=1, dilation=1):
    "kxk convolution with padding"
    if isinstance(kernel_size, tuple):
        kernel_size = np.asarray(kernel_size)
    elif isinstance(kernel_size, int):
        kernel_size = np.asarray((kernel_size, kernel_size))
    else:
        raise ValueError(
            "Parameter 'kernel_size' must be int or tuple. Got {}.".format(
                type(kernel_size)))

    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size

    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size - 1) // 2

    # Conv2d doesn't accept numpy arrays as arguments
    full_padding, kernel_size = tuple(
        [int(elem) for elem in full_padding]), tuple([
            int(elem) for elem in kernel_size])

    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=full_padding,
        dilation=dilation,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        self.conv1 = conv_dil(
            (3, 3), inplanes, planes, stride, dilation=dilation[0])
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_dil((3, 3), planes, planes, dilation=dilation[1])
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(Bottleneck, self).__init__()
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation[1], bias=False,
                               dilation=dilation[1])

        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
