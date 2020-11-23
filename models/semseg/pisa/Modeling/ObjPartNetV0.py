'''
author: -- (-- | at | gmail)

'''
import os
import logging

import torch
import torch.nn as nn

from .ObjPartNetModules.drn import drn_d_54, drn_d_107, drn_d_38
from .ObjPartNetModules.layers import size_splits
from .ObjPartNetModules.resnet import resnet34, resnet101
from .ObjPartNetModules.upsampling_simplified import UpsamplingBilinearlySpoof
from .ObjPartNetModules.vgg import VGGNet

logger = logging.getLogger(__name__)
# pylint: disable=invalid-name, arguments-differ,too-many-locals
# pylint: disable=too-many-instance-attributes,too-many-statements


def _bake_function(f, **kwargs):
    import functools
    return functools.partial(f, **kwargs)


# def _normal_initialization(layer):
#     layer.bias.data.zero_()

def _get_fc_mlp(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3),
        nn.Tanh(),
        nn.Conv2d(in_channels, out_channels, 1)
    )

class ObjPartNetV0(nn.Module):
    ''' Unified network that allows for various
    feature extraction modules to be swapped in and out.
    '''
    def __init__(self, config):
        self._config = config
        model_config = config['model_config']
        arch = model_config['arch']
        output_dims_style = model_config.get('output_dims', 'binary')
        self.separate_heads = model_config.get('separate_heads', False)
        self.binary_classify = config.get('binary_classify', False)
        upsample = model_config.get('upsample', 'bilinear')
        pretrained = model_config.get('pretrained', True)
        assert arch in ['drn', 'drn_54', 'resnet', 'resnet101', 'vgg', 'hourglass']
        super(ObjPartNetV0, self).__init__()
        logger.info(f"Loading ObjPartNetV0 with arch={arch}")

        _output_dims = {'merged':
                            {'416': 9,
                            '65': 10,
                            '2': 5,
                            '34': 2,
                            '98': 9,
                            '72': 0,
                            '258': 6,
                            '427': 3,
                            '45': 9,
                            '207': 9,
                            '368': 0,
                            '113': 10,
                            '308': 2,
                            '23': 6,
                            '397': 0,
                            '25': 11,
                            '347': 9,
                            '59': 9,
                            '284': 13,
                            '31': 0},
                        'sparse':
                                    {'416': 54,
                                    '65': 30,
                                    '2': 19,
                                    '34': 3,
                                    '98': 26,
                                    '72': 1,
                                    '258': 10,
                                    '427': 4,
                                    '45': 47,
                                    '207': 28,
                                    '368': 1,
                                    '113': 30,
                                    '308': 3,
                                    '23': 10,
                                    '397': 1,
                                    '25': 20,
                                    '347': 26,
                                    '59': 47,
                                    '284': 25,
                                    '31': 1},
                        'binary':
                                {'416': 1,
                                    '65': 1,
                                    '2': 1,
                                    '34': 1,
                                    '98': 1,
                                    '72': 1,
                                    '258': 1,
                                    '427': 1,
                                    '45': 1,
                                    '207': 1,
                                    '368': 1,
                                    '113': 1,
                                    '308': 1,
                                    '23': 1,
                                    '397': 1,
                                    '25': 1,
                                    '347': 1,
                                    '59': 1,
                                    '284': 1,
                                    '31': 1}
                        }
        output_dims = _output_dims[output_dims_style]

        # one dimension for each object and part (plus background)
        # to do: in order to train on other datasets, will need to
        # push the bg class to the point file
        mult = 1  # 5 if bbox
        self.mult = mult
        num_classes = 1 + sum(
            [1*mult + output_dims[k]*mult for k in output_dims])
        num_semantic_classes = 1 + len(output_dims)

        if arch == 'resnet':
            step_size = 8
            outplanes = 512
            # This comment block refers to the parameters for the
            # depracated upsampling unit.
            # Net will output y -> list of outputs from important blocks
            # feature_ind denotes which output to concatenate with upsampled
            # 0: conv1
            # 1: layer1
            # Index the feature vector from the base network
            # feature_ind = [0, 1]
            # Give the widths of each feature tensor (i.e dim 2 length)
            # Order is from smallest spatial resolution to largest
            # feature_widths = [64, 64]
            # add skip connections AFTER which upsamples
            # 0 here means BEFORE the first
            # merge_which_skips = set([1, 2])
            # mergedat = {0: (64, 2), 1: (64, 4)}

            # Number of channels at each stage of the decoder
            # upsampling_channels = [512, 128, 64, 32]
            net = resnet34(fully_conv=True,
                           pretrained=pretrained,
                           output_stride=step_size,
                           out_middle=True,
                           remove_avg_pool_layer=True)
        elif arch == 'resnet101':
            step_size = 16
            outplanes = 2048
           
            # Number of channels at each stage of the decoder
            # upsampling_channels = [2048, 128, 64, 32]
            net = resnet101(fully_conv=True,
                           pretrained=pretrained,
                           output_stride=step_size,
                           out_middle=True,
                           remove_avg_pool_layer=True)

        elif arch == 'vgg':
            step_size = 16
            outplanes = 1024
            # feature_ind = [3, 8, 15, 22]
            # feature_widths = [512, 256, 128, 64]
            # merge_which_skips = set([1, 2, 3, 4])
            # upsampling_channels = [1024, 256, 128, 64, 32]
            # mergedat = {15: (512, 8), 8: (256, 4), 1: (128, 2)}
            net = VGGNet()
            raise NotImplementedError(
                "VGGNet architecture not yet debugged")

        elif arch == 'hourglass':
            # step_size = ???
            raise NotImplementedError(
                "Hourglass network architecture not yet implemented")

        elif arch == 'drn':
            step_size = 8
            outplanes = 512
            # feature_ind = [0, 1, 2]
            # feature_widths = [256, 32, 16]
            # merge_which_skips = set([1, 2, 3])
            # feature_ind = [1, 2]
            # feature_widths = [256, 32]
            # merge_which_skips = set([1, 2])
            # upsampling_channels = [512, 128, 64, 32]
            # mergedat = {2: (256, 4), 1: (32, 2)}
            # net = drn_d_107(pretrained=pretrained, out_middle=True)
            net = drn_d_38(pretrained=pretrained, out_middle=True)

        elif arch == 'drn_54':
            step_size = 8
            outplanes = 512
            net = drn_d_54(pretrained=pretrained, out_middle=True)

        # self.inplanes = # num_classes # upsampling_channels[-1]
        # head_outchannels = outplanes // step_size
        # skip_outchannels = [mergedat[k][0] // mergedat[k][1]
        # for k in mergedat]
        # merge_inchannels = head_outchannels + sum(skip_outchannels)
        # self.inplanes = merge_inchannels
        self.inplanes = outplanes
        self.net = net

        # self.fc = nn.Conv2d(self.inplanes, num_classes, 1)
        self.fc = _get_fc_mlp(self.inplanes, num_classes)

        # Randomly initialize the 1x1 Conv scoring layer
        # _normal_initialization(self.fc)

        self.num_classes = num_classes
        self.num_semantic_classes = num_semantic_classes
        output_dims = {int(k): v for k, v in output_dims.items()}
        self.output_dims = output_dims

        # 1 for background, 1 for each object class
        split_tensor = [1]
        # maps each object prediction to its part(s) channels
        op_map = {}
        # Plus the number of part classes present for each cat
        i = 1
        for k in sorted(self.output_dims):
            split_tensor.append(mult)
            i += 1
        j = 1
        for k in sorted(self.output_dims):
            v = self.output_dims[k]
            if v > 0:
                split_tensor.append(v*mult)
                op_map[j] = i
                i += 1
            else:
                op_map[j] = -1
            j += 1

        self.op_map = op_map


        if self.separate_heads:
            self.fc_sem = _get_fc_mlp(self.inplanes, num_semantic_classes)
            if self.binary_classify:
                self.fc_bin = _get_fc_mlp(self.inplanes, 3) # 2 plus a background class
        else:
           
            self.flat_map = {}
            for k, v in op_map.items():
                if v > 0:
                    self.flat_map[k] = v
                    self.flat_map[v] = k
            self.split_tensor = split_tensor
        upsample = UpsamplingBilinearlySpoof(step_size)
        self.decode = upsample

    def _merge_logits(self, input_logits, split_tensor, op_map, binary=False):
        # Add object and part channels to predict a semantic segmentation
        # Assume there is a BG class as well
        splits = size_splits(input_logits, split_tensor, 1)
        # bg, objects, parts = splits[0], splits[1], splits[2:]
        # [bg, obj0, obj1, ..., objN, part0, part1, ..., partN]
        bg, other_data = splits[0], splits[1:]
        if binary:
            other_data = torch.cat(other_data, 1)
            b, d, h, w = other_data.shape
            out = other_data.reshape(b, 2, int(d/2), h, w).sum(2)
            return torch.cat([bg, out], dim=1)
            
        op_data = [torch.sum(part, dim=1, keepdim=True)
                   for part in other_data]
        # the (-1) is since we separate out bg above
        out = []
        for o in sorted(op_map):
            to_add1 = op_data[o-1]
            p = op_map[o]
            if p > 0:
                to_add2 = op_data[p-1]
                out.append(to_add1 + to_add2)
            else:
                out.append(to_add1)

        # out = [op_data[o-1] + op_data[p-1] if p > 0 else
        #        op_data[o-1] for o, p in self.op_map.items()]
        out = torch.cat(out, dim=1)
        return torch.cat([bg, out], dim=1)

    def forward(self, input_d):
        '''
        returns predictions for the object-part segmentation task and
        the semantic segmentation task
            Format:
            x = [background, obj_1, obj_2, ..., parts_1, parts_2, ...]
            out = [background, obj_or_part_1, obj_or_part_2, , ...]
        '''
        x = input_d.get('img')
        logger.debug(f"Rank=[{self._config['rank']}: X.shape =: {x.shape}")
        insize = x.size()[-2:]
        x, _ = self.net(x)    # b x d x h x w
        objpart_logits = self.fc(x) # b x n_classes x h x w
        objpart_logits = self.decode([objpart_logits], insize)
        results = {}
        results['points_logits'] = objpart_logits
        
        if self.separate_heads:
            # Use a separate classification head for objpart and semantic segmentation tasks
            semantic_seg_logits = self.fc_sem(x)
            semantic_seg_logits = self.decode([semantic_seg_logits], insize)
        else:
            # Use sequential classification
            semantic_seg_logits = self._merge_logits(objpart_logits, self.split_tensor, self.op_map)
        results['segmentation_logits'] = semantic_seg_logits

        if self.binary_classify:
            if self.separate_heads:
                binary_logits = self.fc_bin(x)
                binary_logits = self.decode([binary_logits], insize)
            else:
                binary_logits = self._merge_logits(objpart_logits, self.split_tensor, self.op_map, self.binary_classify)
            results['bin_points_logits'] = binary_logits
        return results

    def get_training_parameters(self):
        """
        Return model parameters or grouped parameters to be optimized
        """
        return self.parameters()
