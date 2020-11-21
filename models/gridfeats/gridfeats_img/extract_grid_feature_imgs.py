#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Grid features extraction script.
"""
import argparse
import os
import json
import numpy as np
import cv2
import torch
import tqdm
from fvcore.common.file_io import PathManager

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.evaluation import inference_context
from detectron2.modeling import build_model
import detectron2.data.transforms as T

from grid_feats import (
    add_attribute_config,
    build_detection_test_loader_with_attributes,
)

# A simple mapper from object detection dataset to VQA dataset names
dataset_to_folder_mapper = {}
dataset_to_folder_mapper['coco_2014_train'] = 'train2014'
dataset_to_folder_mapper['coco_2014_val'] = 'val2014'
# One may need to change the Detectron2 code to support coco_2015_test
# insert "coco_2015_test": ("coco/test2015", "coco/annotations/image_info_test2015.json"),
# at: https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/datasets/builtin.py#L36
dataset_to_folder_mapper['coco_2015_test'] = 'test2015'

def extract_grid_feature_argument_parser():
    parser = argparse.ArgumentParser(description="Grid feature extraction")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--dataset", help="name of the dataset", default="coco_2014_train",
                        choices=['coco_2014_train', 'coco_2014_val', 'coco_2015_test'])
    parser.add_argument('--lower', type=int, default=0)
    parser.add_argument('--higher', type=int, default=1)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

def extract_grid_feature_on_dataset(model, image_dir, image_list, dump_folder):
    
    # resize to minimum size 600, maximum size 1000
    aug = T.ResizeShortestEdge(
            [600, 600], 1000
        )
    
    cnt = 0
    for img_id in image_list:
        cnt += 1
        if cnt % 100 == 0:
            print(cnt)
        with torch.no_grad():
            # read in image
            image_path = image_dir + img_id + '.jpg'
            img = cv2.imread(image_path)
            h, w = img.shape[:2]

            new_img = aug.get_transform(img).apply_image(img)
            new_img = torch.as_tensor(new_img.astype("float32").transpose(2, 0, 1))
            inputs = [{"image": new_img, "height": h, "width": w}]
            
            # compute features
            images = model.preprocess_image(inputs)
            features = model.backbone(images.tensor)
            outputs = model.roi_heads.get_conv5_features(features)
            file_name = img_id + '.pt'
            with PathManager.open(os.path.join(dump_folder, file_name), "wb") as f:
                # save as CPU tensors
                torch.save(outputs.cpu(), f)

def do_feature_extraction(cfg, model, image_dir, image_list):
    with inference_context(model):
        # dump_folder = os.path.join(cfg.OUTPUT_DIR, "features")
        dump_folder = 'gridfeats'
        PathManager.mkdirs(dump_folder)
        # data_loader = build_detection_test_loader_with_attributes(cfg, dataset_name)
        extract_grid_feature_on_dataset(model, image_dir, image_list, dump_folder)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_attribute_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # force the final residual block to have dilations 1
    cfg.MODEL.RESNETS.RES5_DILATION = 1
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )

    dataset_path = ''
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    image_path = './images/'
    image_list = list(map(str, list(vqamb.keys())))
    do_feature_extraction(cfg, model, image_path, image_list[args.lower:args.higher])

if __name__ == "__main__":
    args = extract_grid_feature_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
