'''
Rought approximation of VGG from SDD
'''
import torch.nn as nn

# pylint: disable=invalid-name,no-self-use,missing-docstring


class VGGNet(nn.Module):
    def __init__(self, out_indices=None):
        from numbers import Number
        super(VGGNet, self).__init__()
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512,
               512, 'M', 512, 512, 512]
        self.layers = self._get_vgg(cfg, 3)
        self.channels = [c if isinstance(c, Number) else cfg[i-1]
                         for i, c in enumerate(cfg)]
        self.indices = out_indices

    def forward(self, x):
        # pylint: disable=arguments-differ
        y = list()
        for l in self.layers:
            x = l(x)
            y.append(x)
        return x, y

    def _get_vgg(self, cfg, i, out_layers=1024, batch_norm=False):
        layers = []
        in_channels = i
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'C':
                layers += [nn.MaxPool2d(
                    kernel_size=2, stride=2, ceil_mode=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v),
                               nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        conv6 = nn.Conv2d(512, out_layers,
                          kernel_size=3, padding=6, dilation=6)
        # conv7 = nn.Conv2d(1024, out_layers, kernel_size=1)
        layers += [pool5, conv6,
                   nn.ReLU(inplace=True)]
        return layers
