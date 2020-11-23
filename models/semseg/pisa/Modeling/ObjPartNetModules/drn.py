'''
Dilated Residual Networks
Original author: Fisher Yu:
https://github.com/fyu/dilation
And Yu's paper:

@inproceedings{YuKoltun2016,
         author    = {Fisher Yu and Vladlen Koltun},
         title     = {Multi-Scale Context Aggregation by Dilated Convolutions},
         booktitle = {ICLR},
         year      = {2016},
}

Go Tigers!

'''
import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .resnet_layers import (
    BasicBlock,
    Bottleneck)

BatchNorm = nn.BatchNorm2d

# pylint: disable=too-many-instance-attributes,too-many-arguments,
# pylint: disable=missing-docstring,arguments-differ,unused-argument
# pylint: disable=invalid-name


# __all__ = ['DRN', 'drn26', 'drn42', 'drn58']


WEBROOT = 'https://tigress-web.princeton.edu/~fy/drn/models/'

MODEL_URLS = {
    'drn-c-26': WEBROOT + 'drn_c_26-ddedf421.pth',
    'drn-c-42': WEBROOT + 'drn_c_42-9d336e8c.pth',
    'drn-c-58': WEBROOT + 'drn_c_58-0a53a92c.pth',
    'drn-d-22': WEBROOT + 'drn_d_22-4bd2f8ea.pth',
    'drn-d-38': WEBROOT + 'drn_d_38-eebb45f0.pth',
    'drn-d-54': WEBROOT + 'drn_d_54-0e0534ff.pth',
    'drn-d-105': WEBROOT + 'drn_d_105-12b40979.pth'
}


class DRN(nn.Module):
    ''' Dilated Residual Network class.
    '''

    def __init__(self, block, layers, num_classes=1000,
                 channels=(16, 32, 64, 128, 256, 512, 512, 512),
                 out_map=False, out_middle=False, pool_size=28, arch='D',
                 out_indices=None):
        super(DRN, self).__init__()
        self.inplanes = channels[0]
        self.out_map = out_map
        self.out_dim = channels[-1]
        self.out_middle = out_middle
        self.arch = arch
        self.channels = channels
        self.indices = out_indices

        if arch == 'C':
            self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                                   padding=3, bias=False)
            self.bn1 = BatchNorm(channels[0])
            self.relu = nn.ReLU(inplace=True)

            self.layer0 = nn.Sequential(
                self.conv1,
                self.bn1,
                self.relu)
            self.layer1 = self._make_layer(
                BasicBlock, channels[0], layers[0], stride=1)
            self.layer2 = self._make_layer(
                BasicBlock, channels[1], layers[1], stride=2)
        elif arch == 'D':
            self.layer0 = nn.Sequential(
                nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3,
                          bias=False),
                BatchNorm(channels[0]),
                nn.ReLU(inplace=True)
            )

            self.layer1 = self._make_conv_layers(
                channels[0], layers[0], stride=1)
            self.layer2 = self._make_conv_layers(
                channels[1], layers[1], stride=2)

        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)
        self.layer5 = self._make_layer(block, channels[4], layers[4],
                                       dilation=2, new_level=False)
        self.layer6 = None if layers[5] == 0 else \
            self._make_layer(block, channels[5], layers[5], dilation=4,
                             new_level=False)

        if arch == 'C':
            self.layer7 = None if layers[6] == 0 else \
                self._make_layer(BasicBlock, channels[6], layers[6],
                                 dilation=2,
                                 new_level=False, residual=False)
            self.layer8 = None if layers[7] == 0 else \
                self._make_layer(BasicBlock, channels[7], layers[7],
                                 dilation=1,
                                 new_level=False, residual=False)
        elif arch == 'D':
            self.layer7 = None if layers[6] == 0 else \
                self._make_conv_layers(channels[6], layers[6], dilation=2)
            self.layer8 = None if layers[7] == 0 else \
                self._make_conv_layers(channels[7], layers[7], dilation=1)

        # if num_classes > 0:
        #     self.avgpool = nn.AvgPool2d(pool_size)
        #     self.fc = nn.Conv2d(self.out_dim, num_classes, kernel_size=1,
        #                         stride=1, padding=0, bias=True)
        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                n = mod.kernel_size[0] * mod.kernel_size[1] * mod.out_channels
                mod.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(mod, BatchNorm):
                mod.weight.data.fill_(1)
                mod.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    new_level=True, residual=True):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = list()
        layers.append(block(
            self.inplanes, planes, stride, downsample,
            dilation=(1, 1) if dilation == 1 else (
                dilation // 2 if new_level else dilation, dilation),
            residual=residual))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, residual=residual,
                                dilation=(dilation, dilation)))

        return nn.Sequential(*layers)

    def _make_conv_layers(self, channels, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(self.inplanes, channels, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                BatchNorm(channels),
                nn.ReLU(inplace=True)])
            self.inplanes = channels
        return nn.Sequential(*modules)

    def forward(self, x):
        intermediate = list()

        # if self.arch == 'C':
        #     x = self.conv1(x)
        #     x = self.bn1(x)
        #     x = self.relu(x)
        # elif self.arch == 'D':
        #     x = self.layer0(x)
        x = self.layer0(x)

        x = self.layer1(x)
        intermediate.append(x)  # Same
        x = self.layer2(x)
        intermediate.append(x)  # Down 2

        x = self.layer3(x)
        intermediate.append(x)  # down 4

        x = self.layer4(x)
        intermediate.append(x)  # down 8

        # No more downsampling
        x = self.layer5(x)
        intermediate.append(x)

        if self.layer6 is not None:
            x = self.layer6(x)
            intermediate.append(x)

        if self.layer7 is not None:
            x = self.layer7(x)
            intermediate.append(x)

        if self.layer8 is not None:
            x = self.layer8(x)
            intermediate.append(x)

        # if self.out_map:
        #     x = self.fc(x)
        # else:
        #     x = self.avgpool(x)
        #     x = self.fc(x)
        #     x = x.view(x.size(0), -1)
        y = x
        if self.out_middle:
            return y, intermediate
        return y


def drn_c_26(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='C', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(MODEL_URLS['drn-c-26']))
    return model


def drn_c_42(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 1, 1], arch='C', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(MODEL_URLS['drn-c-42']))
    return model


def drn_c_58(pretrained=False, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='C', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(MODEL_URLS['drn-c-58']))
    return model


def drn_d_22(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(MODEL_URLS['drn-d-22']))
    return model


def drn_d_24(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 2, 2], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(MODEL_URLS['drn-d-24']))
    return model


def drn_d_38(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(MODEL_URLS['drn-d-38']))
    return model


def drn_d_40(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 2, 2], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(MODEL_URLS['drn-d-40']))
    return model


def drn_d_54(pretrained=False, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(MODEL_URLS['drn-d-54']))
    return model


def drn_d_56(pretrained=False, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 2, 2], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(MODEL_URLS['drn-d-56']))
    return model


def drn_d_105(pretrained=False, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 23, 3, 1, 1], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(MODEL_URLS['drn-d-105']))
    return model


def drn_d_107(pretrained=False, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 23, 3, 2, 2], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(MODEL_URLS['drn-d-107']))
    return model
