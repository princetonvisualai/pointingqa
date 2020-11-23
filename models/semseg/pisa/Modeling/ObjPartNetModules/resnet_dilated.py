import torch
import torch.nn as nn
# import torchvision.models as models
from .layers import size_splits
from .resnet import resnet34


class Resnet34_8s(nn.Module):
    '''
    The chosen model for Object-Part baseline tests.

    '''

    def __init__(
            self,
            output_dims,
            part_task=False,
            pretrained=True):

        super(Resnet34_8s, self).__init__()
        assert isinstance(output_dims, dict)
        assert len(output_dims) != 0
        assert len(output_dims) == 20

        def _normal_initialization(layer):
            layer.weight.data.normal_(0, 0.01)
            layer.bias.data.zero_()

        # one dimension for each object and part (plus background)
        # TODO: in order to train on other datasets, will need to
        # push the bg class to the point file
        num_classes = 1 + sum([1 + output_dims[k] for k in output_dims])

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_8s = resnet34(fully_conv=True,
                               pretrained=pretrained,
                               output_stride=8,
                               remove_avg_pool_layer=True)

        self.resnet34_8s = resnet34_8s

        # Randomly initialize the 1x1 Conv scoring layer
        resnet34_8s.fc = nn.Conv2d(resnet34_8s.inplanes, num_classes, 1)



        _normal_initialization(self.resnet34_8s.fc)
        self.part_task = part_task
        self.num_classes = num_classes
        # self.num_to_aggregate = num_to_aggregate
        self.output_dims = output_dims
        self.split_tensor = [1, len(self.output_dims)]
        self.split_tensor.extend([v for v in self.output_dims.values()])

    def forward(self, x):

        input_spatial_dim = x.size()[2:]

        x = self.resnet34_8s(x)

        # x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        x = nn.functional.interpolate(
            input=x, size=input_spatial_dim, mode='bilinear')
        if self.part_task:
            # [B, _C_, H, W]
            splits = size_splits(x, self.split_tensor, 1)
            bg, objects, parts = splits[0], splits[1], splits[2:]
            parts = [torch.sum(part, dim=1, keepdim=True) for part in parts]
            parts = torch.cat(parts, dim=1)
            out = objects + parts
            out = torch.cat([bg, out], dim=1)
            # x = [background, obj_1, obj_2, ..., parts_1, parts_2, ...]
            # out = [background, obj_or_part_1, obj_or_part_2, , ...]
            return x, out

        return x
