import logging
import os
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from six.moves import urllib

try:
    from.transforms import get_point_mask
    from .pascal_voc import (
        convert_pascal_berkeley_augmented_mat_annotations_to_png,
        get_augmented_pascal_image_annotation_filename_pairs)
except ImportError:
    from pascal_voc import (
        convert_pascal_berkeley_augmented_mat_annotations_to_png,
        get_augmented_pascal_image_annotation_filename_pairs)
    from transforms import get_point_mask

logger = logging.getLogger(__name__)
_MASKTYPE = {'mode', 'consensus', 'soft'}

# flake8: noqa=E501
# pylint: disable=too-many-instance-attributes, too-few-public-methods, fixme

def prepare_dataset(data_path):
    did_download = download_dataset(data_path)
    extract_dataset(data_path)
    _prepare_dataset(data_path)


#### Download
def download_dataset(data_path):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    did_download_pascal = download_pascal_dataset_if_not_present(data_path)
    did_download_berkeley = download_berkeley_dataset_if_not_present(data_path)
    return did_download_berkeley or did_download_pascal

def download_pascal_dataset_if_not_present(data_path,
                                            tar_name= "VOCtrainval_11-May-2012.tar",
                                            url='http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'):
    logger.info(f"Downloading PASCAL segmentation dataset to {tar_name}")
    return _download_dataset_if_not_present(data_path, tar_name, url)

def download_berkeley_dataset_if_not_present(data_path,
                                            tar_name="benchmark.tgz",
                                            url='http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz'):
    logger.info(f"Downloading BERKELEY segmentation dataset to {tar_name}")
    return _download_dataset_if_not_present(data_path, tar_name, url)
                        
def _download_dataset_if_not_present(data_path, tar_name, url):
    tar_fname = os.path.join(data_path, tar_name)
    if os.path.exists(tar_fname):
        logger.info(f"Dataset file {tar_fname} already exists. Skipping download.")
        return False

    # Add a progress bar for the download
    def _progress(count, block_size, total_size):

        progress_string = "\r>> {:.2%}".format(
            float(count * block_size) / float(total_size))
        sys.stdout.write(progress_string)
        sys.stdout.flush()
    logger.info(f"Downloading segmentation dataset to {tar_fname}")
    urllib.request.urlretrieve(
        url,
        tar_fname,
        _progress)
    return True

## Extracts
def extract_dataset(data_path):
    logger.info(f"Extracting Dataset to {data_path}")
    extract_pascal_dataset_if_not_extracted(data_path)
    extract_berkeley_dataset_if_not_extracted(data_path)

def extract_pascal_dataset_if_not_extracted(data_path, tar_f='VOCtrainval_11-May-2012.tar', extract_dir="."):
    tar_path = os.path.join(data_path, tar_f)
    extract_path = os.path.join(data_path, extract_dir)
    if os.path.exists(os.path.join(extract_path, 'VOCdevkit')):
        return
    _extract_tar(tar_path, extract_path)

def extract_berkeley_dataset_if_not_extracted(data_path, tar_f="benchmark.tgz", extract_dir="."):
    tar_path = os.path.join(data_path, tar_f)
    extract_path = os.path.join(data_path, extract_dir)
    if os.path.exists(os.path.join(extract_path, 'benchmark_RELEASE')):
        return
    _extract_tar(tar_path, extract_path)

def _extract_tar(tar_path, data_path):
     with tarfile.open(tar_path) as tf:
            tf.extractall(path=data_path)

# Prepare
def _prepare_dataset(data_path):
    logger.info(f"Preparing Dataset at {data_path}")
    prepare_pascal_dataset_if_not_prepared(data_path)

def prepare_pascal_dataset_if_not_prepared(data_path, tar_dir="benchmark_RELEASE"):
    root_dir = os.path.join(data_path, tar_dir)
    logger.info("\n Converting .mat files in the Berkeley dataset to pngs")
    convert_pascal_berkeley_augmented_mat_annotations_to_png(root_dir)

def _get_pascal_class_names():
    return [
        'background',  # 0
        'aeroplane',  # 1
        'bicycle',  # 2
        'bird',  # 3
        'boat',  # 4
        'bottle',  # 5
        'bus',  # 6
        'car',  # 7
        'cat',  # 8
        'chair',  # 9
        'cow',  # 10
        'diningtable',  # 11
        'dog',  # 12
        'horse',  # 13
        'motorbike',  # 14
        'person',  # 15
        'potted-plant',  # 16
        'sheep',  # 17
        'sofa',  # 18
        'train',  # 19
        'tvmonitor',  # 20
        'aeroplane_part',  # 21
        'bicycle_part',  # 22
        'bird_part',  # 23
        'boat_part',  # 24
        'bottle_part',  # 25
        'bus_part',  # 26
        'car_part',  # 27
        'cat_part',  # 28
        'chair_part',  # 29
        'cow_part',  # 30
        'diningtable_part',  # 31
        'dog_part',  # 32
        'horse_part',  # 33
        'motorbike_part',  # 34
        'person_part',  # 35
        'potted-plant',  # 36
        'sheep_part',  # 37
        'sofa_part',  # 38
        'train_part',  # 39
        'tvmonitor_part',  # 40
    ]

def _get_index_triplets(point_data, img_anno_pairs, do_filter=False):
    def _get_name(path):
        return Path(path).stem
    for impath, labpath in img_anno_pairs:
        key = _get_name(impath)
        pd = point_data.get(key, None)
        if do_filter and pd is None:
            continue
        yield pd, impath, labpath

def _get_num_labels_dic(matrix, names=[], ignore=255):
    uq = matrix.unique()
    totals = {}
    for val in uq:
        if val == ignore:
            continue
        class_name = names[val]
        totals[class_name] = (matrix == val).sum().item()
    return totals
    
class PascalVOCSegmentationDataset(data.Dataset):

    def __init__(self,
                 config,
                 is_train,
                 point_data,
                 joint_transform=None):
        super().__init__()
        data_path = config['data_path']
        self.mask_type = config.get('mask_type', 'mode')
        assert self.mask_type in _MASKTYPE
        self.binary_points = config.get('binary_classify', False)
        img_anno_pairs = get_augmented_pascal_image_annotation_filename_pairs(
                                                            os.path.join(data_path, 'VOCdevkit'),
                                                            os.path.join(data_path,  'benchmark_RELEASE'))
        if is_train:
            img_anno_pairs = img_anno_pairs[0]
        else:
            img_anno_pairs = img_anno_pairs[1]    
        img_anno_pairs.sort()
        self.data_triplets = list(_get_index_triplets(point_data, img_anno_pairs, not is_train))
        self.joint_transform = joint_transform
        self.class_names = _get_pascal_class_names()
        self.num_objects = len(self.class_names) // 2
        logger.info(f"Creating PascalDataset of length {len(self.data_triplets)}.")

    def __len__(self):
        return len(self.data_triplets)

    def __getitem__(self, index):
        point_annotations, img_path, annotation_path = self.data_triplets[index]
        _img = Image.open(img_path).convert('RGB')
        # TODO: Make format more efficient
        _semantic_target = Image.open(annotation_path)
        _point_target = get_point_mask(
            point_annotations,
            self.mask_type,
            np.array(_semantic_target).shape,
            num_classes=len(self.class_names))
        
        if self.mask_type == 'soft':
            num_sem_classes = len(self.class_names) // 2 + 1
            _semantic_target = np.array(_semantic_target)
            mask = _semantic_target == 255
            _semantic_target[mask] = 0
            targ_shape = _semantic_target.shape + (num_sem_classes,)
            _semantic_target = np.eye(num_sem_classes)[_semantic_target.reshape(-1)].reshape(*targ_shape)
            _semantic_target[mask, :] = 0

        # _point_target = Image.fromarray(_point_target)

        if self.joint_transform is not None:
            _img, _semantic_target, _point_target = self.joint_transform(
                [_img, _semantic_target, _point_target])
        else:
            _img = torch.from_numpy(np.array(_img).copy())
            _semantic_target = torch.from_numpy(np.asarray(_semantic_target).copy()).long()
            _point_target = torch.from_numpy(np.asarray(_point_target).copy()).long()
        num_labels_sem_dic = _get_num_labels_dic(_semantic_target, self.class_names, ignore=255)
        num_labels_sem_dic['semseg_total'] = sum(num_labels_sem_dic.values())
        num_labels_point_dic = _get_num_labels_dic(_point_target, self.class_names, ignore=-1)
        num_labels_point_dic['points_total'] = sum(num_labels_point_dic.values())        

        num_labels =  num_labels_sem_dic['semseg_total']  + num_labels_point_dic['points_total']
        results = {
            'img': _img,
            'segmentation_target': _semantic_target,
            'point_target': _point_target,
            '#SUM#num_labels': num_labels,
            '#SUMDIC#num_semseg_labels': num_labels_sem_dic,
            '#SUMDIC#num_point_labels': num_labels_point_dic,
            'criterion': 'CrossEntropy'
        }

        if self.binary_points:
            _binary_points_target = _point_target.clone()
            mask = _binary_points_target > 0
            _binary_points_target[mask] = (_binary_points_target[mask] // self.num_objects) + 1 # +1 to avoid bg class
            results['binary_point_target'] = _binary_points_target

        return results



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    data_path = os.path.abspath(os.path.join('..', '..', 'data'))
    prepare_dataset(data_path)
    dpath = data_path
    _point_data = {'2007_000032': {}, '2007_000039': {}, '2007_000063': {}, 
                    '2007_000068': {}, '2007_000121': {}, '2007_000170': {}, 
                    '2007_000241': {}, '2007_000243': {}, '2007_000250': {}}
    dataset = PascalVOCSegmentationDataset({'data_path': dpath}, True, _point_data)
