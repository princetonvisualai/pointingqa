# Compute region feature for ground-truth bounding box surrounding point

import yaml
import cv2
import torch
import requests
import numpy as np
import gc
import torch.nn.functional as F
import pandas as pd

import math

import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image

import sys
sys.path.append('./vqa-maskrcnn-benchmark')
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.utils.model_serialization import load_state_dict

import json

class PythiaDetectron:

	def __init__(self):
		self.detection_model = self._build_detection_model()

	# Get the feature representation of the ground truth bounding box (which is passed in as a parameter)
	def get_detectron_gt_features(self, im, bbox_list):
		im, im_scale = self._image_transform(im)
		img_tensor, im_scales = [im], [im_scale]
		current_img_list = to_image_list(img_tensor, size_divisible=32)
		if torch.cuda.is_available():
			current_img_list.to("cuda")
		features = self.detection_model.backbone(current_img_list.tensors.cuda())
		
		bbox_list = (bbox_list * im_scale).cuda()
		bbox_list = BoxList(bbox_list, (im.shape[1], im.shape[0]), mode='xyxy')
		# if torch.cuda.is_available():
		#	bbox_list.to("cuda")

		bbox_list = [bbox_list]
		x, result, detector_losses = self.detection_model.roi_heads(features, bbox_list, None)
		return x, result

	# Transform image
	def _image_transform(self, im):
		im = np.array(im).astype(np.float32)
		im = im[:, :, ::-1]
		im -= np.array([102.9801, 115.9465, 122.7717])
		im_shape = im.shape
		im_size_min = np.min(im_shape[0:2])
		im_size_max = np.max(im_shape[0:2])
		im_scale = float(800) / float(im_size_min)
		# Prevent the biggest axis from being more than max_size
		if np.round(im_scale * im_size_max) > 1333:
		   im_scale = float(1333) / float(im_size_max)
		im = cv2.resize(
		   im,
		   None,
		   None,
		   fx=im_scale,
		   fy=im_scale,
		   interpolation=cv2.INTER_LINEAR
		)
		img = torch.from_numpy(im).permute(2, 0, 1)
		return img, im_scale

	def _build_detection_model(self):

		cfg.merge_from_file('./detectron_model.yaml')
		cfg.freeze()

		model = build_detection_model(cfg)
		checkpoint = torch.load('./detectron_model.pth', 
							  map_location=torch.device("cpu"))

		load_state_dict(model, checkpoint.pop("model"))

		if torch.cuda.is_available():
			model.to("cuda")

		model.eval()
		return model

import json
import os
import argparse

detectron = PythiaDetectron()

if __name__ == '__main__':

	# Iterate through images from low to high (used to parallelize)
	parser = argparse.ArgumentParser()
	parser.add_argument('--lower', type=int)
	parser.add_argument('--higher', type=int)
	args = parser.parse_args()

	print(torch.cuda.is_available())
	
	dataset_path = ''
	with open(dataset_path, 'r') as f:
		dataset = json.load(f)

	detectron = PythiaDetectron()
	
	# For each image in the VQAmb Dataset, compute and store the features
	print(len(dataset))
	cnt = 0
	img_ids = list(dataset.keys())
	

	for img_id in img_ids[args.lower:args.higher]:
		
		cnt = cnt + 1
		if cnt % 10 == 0:
			print(cnt)

		img_path = './images/' + str(img_id) + '.jpg'
		im = Image.open(img_path).convert("RGB")
		w, h = im.size
		
		for qa in dataset[img_id]:
			points = [qa['point']]
			bboxes = [qa['bbox']]
			
			for ind, bbox in enumerate(bboxes):
				
				
				xmin, ymin, xmax, ymax = bbox['x'], bbox['y'], bbox['x']+bbox['w'], bbox['y']+bbox['h']
				bbox_final = torch.Tensor([[xmin, ymin, xmax, ymax]])

				x, y = points[ind]['x'], points[ind]['y']

				fname = './gt_regionfeats/' + str(img_id) + ',' + str(x) + ',' + str(y) + '.pt'
				if os.path.exists(fname): continue
				
				img_feat = detectron.get_detectron_gt_features(im, bbox_final)[0]['fc6']
				torch.save(img_feat, fname)
				
				# might as well also save bounding box
				bbox_fname = './gt_bbox/' + str(img_id) + ',' + str(x) + ',' + str(y) + '.pt'
				torch.save(bbox_final, bbox_fname)
