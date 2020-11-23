'''
transforms.py
author: --
'''
import collections
import json
import numbers
import os
import random

import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.ndimage import zoom
import logging

logger = logging.getLogger(__name__)

# from PIL import Image

# pylint: disable=fixme,invalid-name,missing-docstring


# def get_pascal_object_part_points(points_root, fname):
#     '''
#     Loads the pointwise annotations for the pascal
#     object-part inference task.

#     Arguments:
#       points_root: the root path for the pascal parts
#         dataset folder

#     Returns data, a dictionary of the form:
#     { image id : {
#                   "xCoordinate_yCoordinate": [response1, response2, ...]
#                 }
#     }

#     '''
#     # fname = "pascal_gt.json"
#     # TODO: Check these files...
#     with open(os.path.join(points_root, fname), 'r') as f:
#         # txt = f.readline().strip()
#         parts_per_class = json.loads(f.readline().strip())
#         data = json.loads(f.readline().strip())
#     return data, parts_per_class


def get_valid_circle_indices(arr, center, r):
    ''' Returns the indices of all points of circle (center[1], center[0], r)
    within bounds of array arr.
    '''
    h, w = arr.shape
    i, j = center
    i = min(max(i, 0), h - 1)
    j = min(max(j, 0), w - 1)
    if r > 0:
        istart, istop = max(i - r, 0), min(i + r + 1, h)
        jstart, jstop = max(j - r, 0), min(j + r + 1, w)
        eyes = [y for y in range(istart, istop) for _ in range(jstart, jstop)]
        jays = [x for _ in range(istart, istop) for x in range(jstart, jstop)]
    else:
        eyes = [i]
        jays = [j]
    return eyes, jays


def get_point_mask(point_annotations, mask_type, size, 
                    ignore_index=-1, smooth=False, num_classes=0):

    # Ignore all non-placed points
    point_mask = np.full(size[:2], ignore_index, dtype=np.int32)
    # weights = np.zeros(size, np.float32)
    if not point_annotations:
        if mask_type == 'soft':
            return np.full(size[:2] + (num_classes,), ignore_index, dtype=np.int32)
        return point_mask  # , weights

    # mode: Each annotation is the mode
    # of all responses
    if mask_type == 'mode':
        for point, answers in point_annotations.items():
            coords = point.split("_")
            i, j = int(coords[1]), int(coords[0])
            _answers = np.array([ans for ans in answers if ans >= 0])
            if _answers.size == 0:
                continue
            ans_counts = np.bincount(np.array(_answers))
            modes = np.argwhere(ans_counts == np.amax(ans_counts)).flatten()
            # Choose most common non-ambiguous choice
            # Currently randomly breaks ties.
            # TODO: Develop better logic
            # modes.sort()
            if len(modes) != 1:
                continue
            # np.random.shuffle(modes)
            if smooth:
                inds = get_valid_circle_indices(point_mask, (i, j), 3)
                point_mask[inds] = modes[0]
            else:
                point_mask[i, j] = modes[0]

            if point_mask[i, j] == 0:
                raise RuntimeError(
                    " pointmask 0 here... pascal_part.py line 74")
            # weights[i,j] = 1

    # consensus: only select those points
    # for which the (valid) responses are unanimous.
    # Ignores negative responses.
    elif mask_type == 'consensus':
        for point, answers in point_annotations.items():
            coords = point.split("_")
            i, j = int(coords[1]), int(coords[0])
            _answers = np.array([ans for ans in answers if ans >= 0])
            # Not all responses agree. OR none
            if len(set(_answers)) != 1:
                continue
            # ans_counts = np.argmax(np.bincount(_answers))
            ans_counts = np.bincount(_answers)
            modes = np.argwhere(ans_counts == np.amax(ans_counts)).flatten()

            # Choose most common non-ambiguous choice
            # Currently preferences object over part
            # TODO: Develop better logic
            modes = [m for m in modes if m >= 0]
            if len(modes) != 1:
                continue
            if smooth:
                inds = get_valid_circle_indices(point_mask, (i, j), 3)
                point_mask[inds] = modes[0]
            else:
                point_mask[i, j] = modes[0]

            # weights[i,j] = 1
            if point_mask[i, j] == 0:
                raise RuntimeError(
                    "pointmask 0 here... pascal_part.py line 93")
    elif mask_type == 'soft':
        point_mask = np.full(size[:2] + (num_classes,), ignore_index, dtype=np.int32)
        for point, answers in point_annotations.items():
            coords = point.split("_")
            i, j = int(coords[1]), int(coords[0])
            _answers = np.array([ans for ans in answers if ans >= 0])
            if _answers.size == 0:
                continue
            ans_counts = np.bincount(np.array(_answers), minlength=num_classes)
            ans_counts = ans_counts / ans_counts.sum()
            
            if smooth:
                inds = get_valid_circle_indices(point_mask, (i, j), 3)
                point_mask[inds, :] = ans_counts
            else:
                point_mask[i, j, :] = ans_counts
            
    # weighted: The ground truth annotations
    # are weighted by their ambiguity.
    elif mask_type == 2:
        raise NotImplementedError(
            "mask_type 'weighted' ({}) not implemented".format(mask_type))

    else:
        raise NotImplementedError(
            "mask_type {} not implemented".format(mask_type))

    return point_mask  # , weights

## IMG transforms:


def split_image_into_tiles(input_image, block_rows, block_cols):
    """
    Credit:
    https://stackoverflow.com/questions/16873441/form-a-big-2d-array-from-multiple-smaller-2d-arrays/16873755#16873755
    https://stackoverflow.com/questions/13990465/3d-numpy-array-to-2d/13990648#13990648
    """

    input_rows, input_cols = input_image.shape[:2]

    input_depth = input_image.shape[2] if input_image.ndim == 3 else 0

    # Compute how many blocks will fit along rows and cols axes
    block_cols_num_input = input_cols // block_cols
    block_rows_num_input = input_rows // block_rows

    overall_number_of_blocks = block_rows_num_input * block_cols_num_input

    # Reshaping doesn't change c-arrangement of elements.
    # Reshaping can be looked at like applying ravel() function
    # and then grouping that 1D array into requested shape.

    # So if we form our input image in the following shape (below the comment)
    # we will see that if we swap 1st and 2nd axes (or transpose).
    # Trasposing in this case can be looked at like as if we index first
    # along 2nd and then 1st axis. In case of simple 2D matrix transpose --
    # we traverse elemets down first and right second.

    if input_depth:

        tmp = input_image.reshape((block_rows_num_input,
                                   block_rows,
                                   block_cols_num_input,
                                   block_cols,
                                   input_depth))
    else:

        tmp = input_image.reshape((block_rows_num_input,
                                   block_rows,
                                   block_cols_num_input,
                                   block_cols))

    tmp = tmp.swapaxes(1, 2)

    if input_depth:

        tmp = tmp.reshape(
            (overall_number_of_blocks,
             block_rows,
             block_cols,
             input_depth))

    else:

        tmp = tmp.reshape((overall_number_of_blocks, block_rows, block_cols))

    return tmp


def pad_to_size(input_img, size, fill_label=0):
    """Pads image or array to the size with
    fill_label if the input image is smaller"""
    ohe_enc = False
    if isinstance(input_img, np.ndarray) and len(input_img.shape) == 3:
        ohe_enc = True
        # OHE / Dist encoded array
        input_size = np.array([input_img.shape[0], 
                                input_img.shape[1],
                                input_img.shape[2]])
        size += (input_img.shape[2],)
    elif isinstance(input_img, np.ndarray):
        input_size = np.asarray(input_img.shape)[2:-3:-1]
    else:
        input_size = np.asarray(input_img.size)
    padded_size = np.asarray(size)
    difference = padded_size - input_size
    parts_to_expand = difference > 0
    expand_difference = difference * parts_to_expand
    expand_difference_top_and_left = expand_difference // 2
    expand_difference_bottom_and_right = expand_difference - \
        expand_difference_top_and_left
    # Form the PIL config vector
    pil_expand_array = np.concatenate((expand_difference_top_and_left,
                                       expand_difference_bottom_and_right))
    processed_img = input_img
    # Check if we actually need to expand our image.
    if pil_expand_array.any():
        pil_expand_tuple = tuple(pil_expand_array)
        if ohe_enc:
            arr_padding = ((expand_difference_top_and_left[0],expand_difference_bottom_and_right[0]),
                           (expand_difference_top_and_left[1],expand_difference_bottom_and_right[1]),
                           (0, 0),)
            processed_img = np.pad(
                processed_img,
                pad_width=arr_padding,
                mode='constant',
                constant_values=fill_label)
        elif isinstance(processed_img, np.ndarray):
            arr_padding = (# (0, 0),
                           pil_expand_tuple[1::2],
                           pil_expand_tuple[::2])
            processed_img = np.pad(
                processed_img,
                pad_width=arr_padding,
                mode='constant',
                constant_values=fill_label)
        else:
            processed_img = ImageOps.expand(
                input_img, border=pil_expand_tuple, fill=fill_label)

    return processed_img


def crop_center_numpy(img, crop_size):
    '''Crop to rect in the center
    '''
    crop_width, crop_height = crop_size
    img_height, img_width = img.shape
    start_width = img_width // 2 - (crop_width // 2)
    start_height = img_height // 2 - (crop_height // 2)
    return img[start_height:start_height + crop_height,
               start_width:start_width + crop_width]


def pad_to_fit_tiles_pil(image, tile_size):
    ''' Used if you want multiple tiles/patches
    '''
    original_size_in_pixels = np.asarray(image.size)
    adjusted_size_in_tiles = np.ceil(
        original_size_in_pixels /
        float(tile_size)).astype(
            np.int)
    adjusted_size_in_pixels = adjusted_size_in_tiles * tile_size
    adjusted_img = pad_to_size(image, adjusted_size_in_pixels)
    return adjusted_img, adjusted_size_in_pixels, adjusted_size_in_tiles


def convert_labels_to_one_hot_encoding(labels, number_of_classes):
    ''' returns a tensor [-1, number_of_classes]
    '''

    labels_dims_number = labels.dim()
    # Add a singleton dim -- we need this for scatter
    labels_ = labels.unsqueeze(labels_dims_number)
    # We add one more dim to the end of tensor with the size of
    # 'number_of_classes'
    one_hot_shape = list(labels.size())
    one_hot_shape.append(number_of_classes)
    one_hot_encoding = torch.zeros(one_hot_shape).type(labels.type())

    # Filling out the tensor with ones
    one_hot_encoding.scatter_(dim=labels_dims_number, index=labels_, value=1)

    return one_hot_encoding.byte()


class ComposeJoint(object):
    ''' Apply transformations to both input and target ims/bboxes
    '''
    j = 0

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            self.j += 1
            x = self._iterate_transforms(transform, x)
        return x

    def _iterate_transforms(self, transforms, x):
        """Credit @fmassa:
         https://gist.github.com/fmassa/3df79c93e82704def7879b2f77cd45de
        """
        if isinstance(transforms, collections.Iterable):
            for i, transform in enumerate(transforms):
                x[i] = self._iterate_transforms(transform, x[i])
        else:
            if transforms is not None:
                x = transforms(x)

        return x


class RandomHorizontalFlipJoint(object):

    def __call__(self, inputs):
        # Perform the same flip on all of the inputs
        if random.random() < 0.5:
            # h x w
            image_width = inputs[0].size[0]
            out = []
            for single_input in inputs:
                si = single_input
                if isinstance(si, dict):
                    for k in si:
                        # [x0, y0, w, h]
                        bb = si[k]
                        bb[0] = image_width - bb[0]
                elif isinstance(si, np.ndarray):
                    try:
                        ax = list(si.shape).index(image_width)
                    except ValueError:
                        logger.warning("Image and mask sizes not compatible"
                              "for flipping (transforms.py)")
                        ax = 1
                    si = np.flip(single_input, axis=ax).copy()
                else:
                    si = ImageOps.mirror(single_input)
                out.append(si)
            # return [ImageOps.mirror(single_input) for single_input in inputs]
            return out
        return inputs


class RandomScaleJoint(object):

    def __init__(
            self,
            low,
            high,
            interpolations=(
                Image.BILINEAR,
                Image.NEAREST)):
        self.low = low
        self.high = high
        self.interpolations = interpolations
        import warnings
        warnings.filterwarnings('ignore', '.*output shape of zoom.*')

    def __call__(self, inputs):
        ratio = random.uniform(self.low, self.high)
        height, width = inputs[0].size[0], inputs[0].size[1]
        new_height, new_width = (int(ratio * height), int(ratio * width))

        def resize_input(x, interpolation):
            if isinstance(x, np.ndarray):
                _, w, h = x.shape
                array_zoom = (1, new_width / float(w), new_height / float(h))
                si = zoom(x, array_zoom, order=interpolation)
                check1 = si.shape[1] == new_width
                check2 = si.shape[2] == new_height
                if not check1 or not check2:
                    logger.warning("zoom resize incorrect:\t{}, {}".format(
                        check1, check2))
                return si
            return x.resize((new_height, new_width), interpolation)
        return [resize_input(inp, x) for inp, x in zip(
            inputs, self.interpolations)]


class FixedScaleJoint(object):
    def __init__(
            self,
            outsize,
            interpolations=(
                Image.BILINEAR,
                Image.NEAREST)):
        if isinstance(outsize, int):
            self.outsize = (outsize, outsize)
        else:
            self.outsize = outsize
        self.interpolations = interpolations
        import warnings
        warnings.filterwarnings('ignore', '.*output shape of zoom.*')

    def __call__(self, inputs):
        # ratio = random.uniform(self.low, self.high)
        # height, width = inputs[0].size[0], inputs[0].size[1]
        # new_height, new_width = (int(ratio * height), int(ratio * width))
        new_height, new_width = self.outsize[0], self.outsize[1]

        def resize_input(x, interpolation):
            if isinstance(x, np.ndarray):
                _, w, h = x.shape
                array_zoom = (1, new_width / float(w), new_height / float(h))
                si = zoom(x, array_zoom, order=interpolation)
                check1 = si.shape[1] == new_width
                check2 = si.shape[2] == new_height
                if not check1 or not check2:
                    logger.warning("zoom resize incorrect:\t{}, {}".format(
                        check1, check2))
                return si
            return x.resize((new_height, new_width), interpolation)
        return [resize_input(inp, x) for inp, x in zip(
            inputs, self.interpolations)]


def fixed_scale_valpoints(dic, insize, outsize):
    # pylint: disable=too-many-locals
    ''' Currently, validation points are held in
         a 1D index. This func converts this to a 1D
         index in the new image space.
            h, w = insize
            newh, neww = outsize
    '''
    def oned22d(p, w):
        i = int(p / w)
        j = int(p % w)
        return i, j

    def twod21d(p, w):
        i, j = p
        return i * w + j
    # smaller_edge_out = outsize[0]
    # smaller_edge = min(insize)
    # outsize = smaller_edge_out / float(smaller_edge)
    # outsize = [int(outsize*i + 0.5) for i in insize]
    dic_ = {}
    for objpart, dat in dic.items():
        dic_[objpart] = {}
        for cat, points in dat.items():
            newpoints = []
            for point in points:
                newp = oned22d(point, insize[1])
                inew = (float(newp[0])/insize[0])*outsize[0]
                inew = min(int(inew+0.5), outsize[0])
                jnew = (float(newp[1])/insize[1])*outsize[1]
                jnew = min(int(jnew+0.5), outsize[1])
                newp = twod21d((inew, jnew), outsize[1])
                newpoints.append(newp)
            dic_[objpart][cat] = newpoints
    logger.warning(f"sizes: {insize}, {outsize}, {dic}, {dic_}")
    return dic_


class RandomCropJoint(object):

    def __init__(self, crop_size, pad_values=(0, 255)):
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size
        self.pad_values = pad_values

    def __call__(self, inputs):
        # Assume that the first input is an image
        # (not super robust...)
        # used for bbox scaling
        # pylint: disable=too-many-locals
        difference = np.asarray(self.crop_size) - np.asarray(inputs[0].size)
        difference = difference * (difference > 0)
        expand_difference_top_and_left = difference // 2

        def padd_input(x, pad_value):
            if isinstance(x, dict):
                for k in x:
                    bb = x[k]
                    bb[0] += expand_difference_top_and_left[0]
                    bb[1] += expand_difference_top_and_left[1]
                return x

            return pad_to_size(x, self.crop_size, pad_value)

        padded_inputs = [padd_input(inp, v)
                         for inp, v in zip(inputs, self.pad_values)]

        # We assume that inputs were of the same size before padding.
        # So they are of the same size after the padding
        w, h = padded_inputs[0].size
        th, tw = self.crop_size

        if w == tw and h == th:
            return padded_inputs

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        outputs = []
        for single_input in padded_inputs:
            if isinstance(single_input, dict):
                si = single_input
                for k in si:
                    bb = si[k]
                    bb[0] -= x1
                    bb[1] -= y1
                outputs.append(si)
            elif isinstance(single_input, np.ndarray):
                si = single_input
                si = si[:, y1:y1+th, x1:x1+tw].copy()
                outputs.append(si)
            else:
                outputs.append(single_input.crop((x1, y1, x1 + tw, y1 + th)))
        # outputs = [single_input.crop(
        #     x1, y1, x1 + tw, y1 + th) for single_input in padded_inputs]
        return outputs


class CropOrPad(object):

    def __init__(self, output_size, fill=0):

        self.fill = fill
        self.output_size = output_size

    def __call__(self, x):
        x_size = x.size
        x_position = (np.asarray(self.output_size) // 2) - \
            (np.asarray(x_size) // 2)
        output = Image.new(mode=x.mode,
                           size=self.output_size,
                           color=self.fill)
        output.paste(x, box=tuple(x_position))
        return output


class ResizeAspectRatioPreserve(object):

    def __init__(self, greater_side_size, interpolation=Image.BILINEAR):

        self.greater_side_size = greater_side_size
        self.interpolation = interpolation

    def __call__(self, x):
        w, h = x.size
        if w > h:
            ow = self.greater_side_size
            oh = int(self.greater_side_size * h / w)
            return x.resize((ow, oh), self.interpolation)
        oh = self.greater_side_size
        ow = int(self.greater_side_size * w / h)
        return x.resize((ow, oh), self.interpolation)


class Copy(object):

    def __init__(self, number_of_copies):
        self.number_of_copies = number_of_copies

    def __call__(self, input_to_duplicate):
        # Inputs can be of different types: numpy, torch.Tensor, PIL.Image
        duplicates_array = []
        if isinstance(input_to_duplicate, torch.Tensor):

            for _ in range(self.number_of_copies):
                duplicates_array.append(input_to_duplicate.clone())
        else:

            for _ in range(self.number_of_copies):
                duplicates_array.append(input_to_duplicate.copy())

        return duplicates_array


# Helper functions for reverse() method (below)
def _squeeze_for_tensor_list(list_of_tensors, dim):
    return [x.squeeze(dim) for x in list_of_tensors]


def _squeeze_for_2D_tensor_list(list2D_of_tensors, dim):
    return [_squeeze_for_tensor_list(x, dim) for x in list2D_of_tensors]


# Assumed to be run on torch.Tensor
class Split2D(object):
    """
    Splits the Tensor into 2D tiles along given two dimensions,
    and stacks them along specified new dimension. Mainly used to
    split input 2D image into nonintersecting tiles and stack them
    along batch dimension. Can be used when the whole image doesn't fit
    into the available GPU memory.
    """

    def __init__(self,
                 split_block_sizes=(128, 128),
                 split_dims=(1, 2),
                 stack_dim=0):

        self.split_block_sizes = split_block_sizes
        self.split_dims = split_dims
        self.stack_dim = stack_dim

    def __call__(self, tensor_to_split):

        split_2d = []

        split_over_first_dim = tensor_to_split.split(self.split_block_sizes[0],
                                                     dim=self.split_dims[0])

        for current_first_dim_split in split_over_first_dim:

            split_2d.extend(
                current_first_dim_split.split(
                    self.split_block_sizes[1],
                    dim=self.split_dims[1]))

        res = torch.stack(split_2d, dim=self.stack_dim)

        return res

    def reverse(self, tensor_to_unsplit, dims_sizes):

        # First we get separate rows
        separate_rows = torch.split(tensor_to_unsplit,
                                    split_size=dims_sizes[1],
                                    dim=self.stack_dim)

        # Split each rows into separate column elements
        tensor_list_2D = [torch.split(
            x, split_size=1, dim=self.stack_dim) for x in separate_rows]
        # tensor_list_2D = list(map(lambda x: torch.split(
        #     x, split_size=1, dim=self.stack_dim), separate_rows))

        # Remove singleton dimension, so that we can use original
        # self.split_dims
        tensor_list_2D = _squeeze_for_2D_tensor_list(
            tensor_list_2D, self.stack_dim)

        concatenated_columns = [torch.cat(
            x, dim=self.split_dims[1]) for x in tensor_list_2D]

        # concatenated_columns = list(
        #     map(lambda x:
        #         torch.cat(x, dim=self.split_dims[1]), tensor_list_2D))

        unsplit_original_tensor = torch.cat(
            concatenated_columns, dim=self.split_dims[0])

        return unsplit_original_tensor


# Below, functions adapted from
# https://github.com/fyu/drn/blob/master/data_transforms.py
def pad_reflection(image, top, bottom, left, right):
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return image
    h, w = image.shape[:2]
    next_top = next_bottom = next_left = next_right = 0
    if top > h - 1:
        next_top = top - h + 1
        top = h - 1
    if bottom > h - 1:
        next_bottom = bottom - h + 1
        bottom = h - 1
    if left > w - 1:
        next_left = left - w + 1
        left = w - 1
    if right > w - 1:
        next_right = right - w + 1
        right = w - 1
    new_shape = list(image.shape)
    new_shape[0] += top + bottom
    new_shape[1] += left + right
    new_image = np.empty(new_shape, dtype=image.dtype)
    new_image[top:top+h, left:left+w] = image
    new_image[:top, left:left+w] = image[top:0:-1, :]
    new_image[top+h:, left:left+w] = image[-1:-bottom-1:-1, :]
    new_image[:, :left] = new_image[:, left*2:left:-1]
    new_image[:, left+w:] = new_image[:, -right-1:-right*2-1:-1]
    return pad_reflection(new_image, next_top, next_bottom,
                          next_left, next_right)


def pad_constant(image, top, bottom, left, right, value):
    # pylint: disable=too-many-arguments
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return image
    h, w = image.shape[:2]
    new_shape = list(image.shape)
    new_shape[0] += top + bottom
    new_shape[1] += left + right
    new_image = np.empty(new_shape, dtype=image.dtype)
    new_image.fill(value)
    new_image[top:top+h, left:left+w] = image
    return new_image


def pad_image(mode, image, top, bottom, left, right, value=0):
    ''' Pads image. Only works for PIL images though.
    '''
    # pylint: disable=too-many-arguments
    if mode == 'reflection':
        return Image.fromarray(
            pad_reflection(np.asarray(image), top, bottom, left, right))
    elif mode == 'constant':
        return Image.fromarray(
            pad_constant(np.asarray(image), top, bottom, left, right, value))
    else:
        raise ValueError('Unknown mode {}'.format(mode))


class RandomRotate(object):
    """Rotates the given PIL.Image at a random angle +/- angle
    """
    # pylint: disable=no-self-use
    def __init__(self, angle):
        from scipy.ndimage import interpolation
        rotate = interpolation.rotate
        # import interpolation.rotate as rotate
        self.rotate = rotate
        self.angle = angle

    def rotate_pilim(self, image, label):
        w, h = image.size
        # p = max((h, w))
        if label is not None:
            label = pad_image('constant', label, h, h, w, w, value=255)
        #    label = label.rotate(angle, resample=Image.NEAREST)
            label = label.crop((w, h, w + w, h + h))
        image = pad_image('reflection', image, h, h, w, w)
        # image = image.rotate(angle, resample=Image.BILINEAR)
        image = image.crop((w, h, w + w, h + h))
        return image, label

    def __call__(self, image, label=None, *args):
        # pylint: disable=keyword-arg-before-vararg
        assert label is None or image.size == label.size

        #  angle = random.randint(0, self.angle * 2) - self.angle
        if isinstance(image, np.ndarray):
            raise RuntimeError("Not yet implemented for numpy arrays")
        image, label = self.rotate_pilim(image, label)
        return image, label
