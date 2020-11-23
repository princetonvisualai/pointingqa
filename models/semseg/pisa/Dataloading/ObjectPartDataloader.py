import json
from json.decoder import JSONDecodeError
import logging
import os
import re

import numpy as np
import torch
import torchvision
from torch._six import container_abcs, int_classes, string_classes
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader

try:
    from .PascalDataset import PascalVOCSegmentationDataset, prepare_dataset
    from .transforms import (ComposeJoint, RandomCropJoint,
                             RandomHorizontalFlipJoint)
except:
    from PascalDataset import PascalVOCSegmentationDataset, prepare_dataset
    from transforms import (ComposeJoint, RandomCropJoint,
                            RandomHorizontalFlipJoint)

np_str_obj_array_pattern = re.compile(r'[SaUO]')
logger = logging.getLogger(__name__)

def _numpy_2_torch(x):
        return torch.from_numpy(np.asarray(x).copy()).long()

def _collate(batch, key=None):
        r"""Puts each data field into a tensor with outer dimension batch size"""
        special_keys = ['#SUM#', '#SUMDIC#', '#DONOTCOLLATE#']
        elem = batch[0]
        elem_type = type(elem)
        if key is not None:
            if key.startswith(special_keys[0]):
                return sum(batch)
            if key.startswith(special_keys[1]):
                _key = key[len(special_keys[1]):]
                sumdic = {}
                for d in batch:
                    for k, v in d.items():
                        if k not in sumdic:
                            sumdic[k] = 0
                        sumdic[k] += v
                return sumdic
            if key.startswith(special_keys[2]):
                return batch
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(elem_type)

                return _collate([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int_classes):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            results = {}
            for key in elem:
                if key.startswith('#'):
                    results[key.split('#')[-1]] = _collate([d[key] for d in batch], key)
                else:
                    results[key] = _collate([d[key] for d in batch])
            return results
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(_collate(samples) for samples in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError('each element in list of batch should be of equal size')
            transposed = zip(*batch)
            return [_collate(samples) for samples in transposed]

        raise TypeError(elem_type)


def _get_split_data(config, label):
    splitfile_names = {
        'train': 'ppd_train_images.npy',
        # 'val':   'ppd_val_images.npy',
        # 'dev':   'ppd_val_images.npy',
        'val':   'ppd_combined_val_images.npy', # combine val and test sets into  single larger val set 
        'dev':   'ppd_combined_val_images.npy', 
        # 'test':  'ppd_test_images.npy',
        'test':  'ppd_final_images.npy'
    }
    jsonfile_name = config.get('points_file_name', 'pascal_gt_clean.json')
    logger.info(f"Loading points from {jsonfile_name}")
    datafile_path = os.path.join(config['data_path'], jsonfile_name)
    try:
        with open(datafile_path) as f:
            points_data = json.load(f)
    except JSONDecodeError as err:
        logger.error(f"Error decoding {datafile_path}")
        raise
    logger.info(f"Loading points from {jsonfile_name}")
    splitfile_name = os.path.join(config['data_path'], splitfile_names[label])
    logger.info(f"Loading splitfile from {splitfile_name}")
    labels_to_keep = set(np.load(splitfile_name).tolist())
    split_type = config.get('label_selection_criterion', 'all') # all, ambiguous, or unambiguous
    
    def _check_labels(dic):
        if len(dic) == 0:
            return False
        return sum(len(v)for v in dic.values()) > 0
    
    def _select_labels(annots):
        if split_type == 'all':
            return annots
        results = {}
        for pointix, labels in annots.items():
            if split_type == 'ambiguous':
                if all([label == labels[0] for label in labels]):
                    continue
                else:
                    results[pointix] = labels
            elif split_type == 'unambiguous':
                if all([label == labels[0] for label in labels]):
                    results[pointix] = labels
                else:
                    continue
        return results
    num_points = 0
    results = {}
    for k, v in points_data.items():
        if k not in labels_to_keep or not _check_labels(v):
            continue
        labels = _select_labels(v)
        if not labels:
            continue
        results[k] = labels
        num_points += len(labels)

    logger.info(f"Loaded #{num_points} annotations for split {label}")
    return results
    # return {k: _select_labels(v) for k, v in points_data.items() if k in labels_to_keep and _check_labels(v)}


def ObjectPartDataFactory(config, label, world_size=1, rank=1, seed=None, **kwargs):
    '''
    Emit a dataloader for the Object Part inference task using the specified
    config.
    '''
    data_root = config['data_path']
    labels = _get_split_data(config, label)
    batch_size = config['batch_size']
    num_workers = config.get('num_workers', 2)
    is_train = label == 'train'
    insize = config.get('insize', 512) # transformed image size
    joint_transform = None

    if is_train:
        joint_transform = ComposeJoint(
        [
            RandomHorizontalFlipJoint(),
            RandomCropJoint(crop_size=(insize, insize), pad_values=[
                0, 255, -1]),
            [torchvision.transforms.ToTensor(), None, None],
            [torchvision.transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), None, None],
            # convert labels to tensors
            [None, torchvision.transforms.Lambda(
                _numpy_2_torch),
             # Point Labels
             torchvision.transforms.Lambda(
                 _numpy_2_torch)]
        ])
    else:
        joint_transform = ComposeJoint(
        [
            [torchvision.transforms.ToTensor(), None, None],
            [torchvision.transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), None, None],
            # convert labels to tensors
            [None, torchvision.transforms.Lambda(
                _numpy_2_torch),
             # Point Labels
             torchvision.transforms.Lambda(
                 _numpy_2_torch)]
        ])

    prepare_dataset(data_root)
    dataset = PascalVOCSegmentationDataset(config, is_train, labels, joint_transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=is_train) if world_size > 1 else (RandomSampler(dataset) if is_train else SequentialSampler(dataset))
    logger.debug(f"RANK={rank}/{world_size}: Initialized Sampler: {sampler}")
    dataloader = DataLoader(dataset=dataset,
                            sampler=sampler,
                            batch_size=batch_size if is_train else 1,
                            collate_fn=_collate,
                            num_workers=num_workers,
                            pin_memory=True)

    return dataloader



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    data_path = os.path.abspath(os.path.join('..', '..', 'data'))
    prepare_dataset(data_path)
    config = {
        'data_path': data_path,
        'batch_size': 10
    }
    logger.info(f"Creating ObjectPartDataFactory with config: {config}")
    _dataset = ObjectPartDataFactory(config, 'train')
    for i, batch in enumerate(_dataset):
        sizes = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                sizes[k] = v.shape
            else:
                sizes[k] = v
        logger.info(f"[{i}]: {sizes}")
        if i > 5:
            break
    import pdb; pdb.set_trace()
