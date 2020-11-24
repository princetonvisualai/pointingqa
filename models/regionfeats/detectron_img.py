# Compute region features for entire image (no point filtering)

import yaml
import cv2
import torch
import requests
import numpy as np
import gc
import torch.nn.functional as F
import pandas as pd
import os

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

	# Transform image
	def _image_transform(self, image_path, bbox=None):

		img = Image.open(image_path)
		if img.mode != "RGB":
			img = img.convert("RGB")
		if bbox != None:
			img = img.crop(bbox)
		im = np.array(img).astype(np.float32)
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

		cfg.merge_from_file('detectron_model.yaml')
		cfg.freeze()

		model = build_detection_model(cfg)
		checkpoint = torch.load('detectron_model.pth', 
							  map_location=torch.device("cpu"))

		load_state_dict(model, checkpoint.pop("model"))

		if torch.cuda.is_available():
			model.to("cuda")

		model.eval()
		return model

	def run_detectron(self, image_path):
		im, im_scale = self._image_transform(image_path)
		img_tensor, im_scales = [im], [im_scale]
		current_img_list = to_image_list(img_tensor, size_divisible=32)

		if torch.cuda.is_available():
			current_img_list = current_img_list.to("cuda")

		with torch.no_grad():
			output = self.detection_model(current_img_list)

		return output, im_scales

	
	def _process_feature_extraction(self, output, im_scales, feat_name='fc6'):
		
		bboxes, scores, feats = output[0]['proposals'][0].bbox, output[0]['scores'], output[0]['fc6']
		scores = F.softmax(scores, dim=1)

		feat_list = []

		dets = bboxes / im_scales[0]

		max_conf = torch.zeros((scores.shape[0])).cuda()

		for cls_ind in range(1, scores.shape[1]):
			cls_scores = scores[:, cls_ind]
			keep = nms(dets, cls_scores, 0.5)
			max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
					   cls_scores[keep],
					   max_conf[keep])

		keep_boxes = torch.argsort(max_conf, descending=True)[:100]
		return feats[keep_boxes], bboxes[keep_boxes]

import json
import os
import argparse

detectron = PythiaDetectron()

if __name__ == '__main__':

	# Iterate through images from low to high (used to parallelize)
	parser = argparse.ArgumentParser()
	parser.add_argument('--lower', type=int, default=0)
	parser.add_argument('--higher', type=int, default=1)
	args = parser.parse_args()
	
	objpart_path = ''
	with open(objpart_path, 'r') as f:
		objpart = json.load(f)

	detectron = PythiaDetectron()
	
	# For each image in the VQAmb Dataset, compute and store the features
	print(len(objpart))
	cnt = 0
	img_ids = list(objpart.keys())

	'''
	Possible modes for now:
	1. 'post_bbox_all': take all bounding boxes that contain the point.
	2. 'post_bbox_all_excl': take all bounding boxes that contain the point and don't contain another point
	3. 'post_bbox_topn': take all bounding boxes from top N that contain the point
	4. 'bbox_crop': crop image to ground truth bounding box 
	5. 'point_crop': crop image to max half the image size surrounding the point
	'''

	for img_id in img_ids[args.lower:args.higher]:
		
		cnt = cnt + 1
		if cnt % 10 == 0:
			print(cnt)

		if objpart[img_id][0]['div'] != 'test':
			continue
			
		fname = str(img_id) + '.pt'
		if os.path.exists(fname): continue
		
		
		img_path = ''
		output, im_scale = detectron.run_detectron(img_path)
		img_feat, img_bboxes = detectron._process_feature_extraction(output, im_scale)
		
		torch.save(img_feat, fname)

		bbox_fname = ''
		# if os.path.exists(bbox_fname): continue

		torch.save(img_bboxes, bbox_fname)
		
