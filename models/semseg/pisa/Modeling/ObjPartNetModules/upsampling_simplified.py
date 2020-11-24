''' Flexible upsampling and merging module that allows for the use
of transposed convolutions or bilinear upsampling. Allows for the
optional inclusion of skip layers as in the original FCN.
'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# pylint: disable=too-many-arguments, too-many-locals, unidiomatic-typecheck,
# pylint: disable=invalid-name,arguments-differ,too-many-instance-attributes
# pylint: disable=too-many-function-args, redefined-builtin


class BilinearUpsample(nn.Module):
    ''' Allow for dynamic "size" args at runtime.
    though probably not adviseable...
    '''
    def __init__(self, scale_factor=None, mode='bilinear'):
        super(BilinearUpsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, input, size=None):
        if size:
            if isinstance(input, list):
                return [F.interpolate(
                    x,
                    size=size,
                    # scale_factor=self.scale_factor,
                    mode=self.mode) for x in input]
            return F.interpolate(input,
                              size=size,
                              # scale_factor=self.scale_factor,
                              mode=self.mode)

        if isinstance(input, list):
            return [F.interpolate(
                x,
                # size=size,
                scale_factor=self.scale_factor,
                mode=self.mode) for x in input]
        return F.interpolate(input,
                          # size=size,
                          scale_factor=self.scale_factor,
                          mode=self.mode)


class UpsamplingBilinearlySpoof(nn.Module):
    ''' Fits in to replace the upsampling module below.
    '''
    def __init__(self, stride):
        super(UpsamplingBilinearlySpoof, self).__init__()
        self.stride = stride

    def forward(self, x, size=None):
        x = x[0]
        if size is None:
            return nn.functional.interpolate(
                x, scale_factor=self.stride, mode='bilinear')
        return nn.functional.interpolate(x, size=size, mode='bilinear')


class Upsampling(nn.Module):
    ''' Defines a decoder.
    '''
    def __init__(
            self,
            stride,
            inchannels,
            outchannels,
            mergedat):
        '''
            args:
                :parameter ``stride``: the outstride of the network or total
                downsampling rate
                :parameter ``inchannels``: int -> number of channels in the
                primary network
                :parameter ``channels``: int -> number of channels to output
                at original size
                :parameter ``mergedat``: dict of
                {feature_ind : (inchannels, stride), ...}
        '''
        super(Upsampling, self).__init__()

        self.conv1up = make_separated_transposed_blockchain(inchannels, stride)
        head_outchannels = inchannels // stride
        skip_outchannels = [mergedat[k][0] // mergedat[k][1] for k in mergedat]
        merge_inchannels = head_outchannels + sum(skip_outchannels)
        self.merge_inchannels = merge_inchannels
        self.head_outchannels = head_outchannels
        self.skip_outchannels = skip_outchannels
        self.outchannels = outchannels
        self.mergeout = nn.Conv2d(merge_inchannels, outchannels, 1)
        skip_chains = []
        skip_chains_inds = []
        for k, (inc, in_stride) in mergedat.items():
            skip_chains.append(
                make_separated_transposed_blockchain(inc, in_stride))
            skip_chains_inds.append(k)
        self.skip_chains = nn.ModuleList(skip_chains)
        self.skip_chain_inds = skip_chains_inds

    def forward(self, x):
        x, low, _ = x

        x = self.conv1up(x)
        features = [x]
        for ind, lay in zip(self.skip_chain_inds, self.skip_chains):
            features.append(lay(low[ind]))
        x = torch.cat(features, dim=1)
        # x = nn.functional.interpolate(x, size=outsize, mode='bilinear')
        # x = self.mergeout(x)
        return x


def make_separated_transposed_blockchain(inchannels, stride):
    ''' Make a chain of separated transposed convolutional blocks
    '''
    num_up = math.log(stride, 2)
    assert num_up % 1 == 0, 'stride must be a power of 2'
    num_up = int(num_up)
    _inchannels = inchannels
    layers = []
    for _ in range(num_up):
        layers.append(SeparatedTransposedBlock(_inchannels))
        _inchannels = _inchannels // 2
    return nn.Sequential(*layers)


class SeparatedTransposedBlock(nn.Module):
    ''' From "Devil is in the Decoder"
    Implements a residual separated transposed convolutional block
    to double the spatial resolution while halving the
    width.
    '''
    def __init__(self, inchannels, outchannels=None):
        super(SeparatedTransposedBlock, self).__init__()
        # self.convres = nn.Conv2d(inchannels, inchannels*2, 1)

        self.convup = nn.Sequential(
            *[nn.ConvTranspose2d(
                inchannels, inchannels,
                3, stride=2, padding=1,
                output_padding=1, groups=inchannels),
              nn.Conv2d(inchannels, outchannels, 1)])

    def _upsample_residual(self, x):
        ''' Try to upsample a residual
        '''
        x = self.convres(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
        n, c, h, w = x.size()
        res = x.view(n, c//4, 4, h, w).sum(2)
        return res

    def forward(self, x):
        # res = self._upsample_residual(x)
        x = self.convup(x)
        return x  # + res
