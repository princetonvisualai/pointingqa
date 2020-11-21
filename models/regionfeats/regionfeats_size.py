# Get region features based on size (smallest bbox, largest bbox, etc.)

import yaml
import cv2
import torch
import requests
import numpy as np
import gc
import torch.nn.functional as F
import pandas as pd

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

	def run_detectron(self, image_path):
		im, im_scale = self._image_transform(image_path)
		img_tensor, im_scales = [im], [im_scale]
		current_img_list = to_image_list(img_tensor, size_divisible=32)

		if torch.cuda.is_available():
			current_img_list = current_img_list.to("cuda")

		with torch.no_grad():
			output = self.detection_model(current_img_list)

		return output, im_scale

	# Take all bounding boxes porposals surrounding a point
	def _process_bbox(self, output, point, points, im_scale, feat_name, feat_type='small'):
		bboxes, scores, feat = output[0]['proposals'][0].bbox, output[0]['scores'], output[0][feat_name]

		bboxes = bboxes / im_scale

		scores = F.softmax(scores, dim=1)

		xmin, ymin, xmax, ymax = bboxes.split(1, dim=1)
		filt_pt = ((point['x'] >= xmin) & (point['x'] <= xmax) & (point['y'] >= ymin) & (point['y'] <= ymax)).flatten()

		# Remove bboxes that also contain points from another answer
		if points != None:
			for pt in points:
				if pt['ans'] != point['ans']:
					filt_ignore = ((pt['x'] >= xmin) & (pt['x'] <= xmax) & (pt['y'] >= ymin) & (pt['y'] <= ymax)).flatten()
					filt_pt = filt_pt * (~filt_ignore)
	    
		bboxes, scores, feat = bboxes[filt_pt], scores[filt_pt], feat[filt_pt]

		# check if no bounding boxes contain that point																				  
		if bboxes.nelement() == 0:
			return torch.zeros(100, 2048), torch.zeros(100, 4)

		# compute bbox areas, get feature of smallest bounding box
		if feat_type == 'small':
			xmin, ymin, xmax, ymax = bboxes.split(1, dim=1)
			bbox_area = ((xmax - xmin) * (ymax - ymin)).flatten()
			sorted_boxes = torch.argsort(bbox_area)[0]

		# compute bbox areas, get feature of largest bounding box																	       
		if feat_type == 'large':
			xmin, ymin, xmax, ymax = bboxes.split(1, dim=1)
			bbox_area = ((xmax - xmin) * (ymax - ymin)).flatten()
			sorted_boxes = torch.argsort(bbox_area, descending=True)[0]

		# get feature of bounding box closes to average area
		elif feat_type == 'avg':
			AVG_AREA = 9970
			xmin, ymin, xmax, ymax = bboxes.split(1, dim=1)
			bbox_area = ((xmax - xmin) * (ymax - ymin)).flatten()
			diff_area = torch.square(bbox_area - AVG_AREA)
			sorted_boxes = torch.argsort(diff_area)[0]

		elif feat_type == 'avgwh':
			AVG_WIDTH = 82
			AVG_HEIGHT = 75
			xmin, ymin, xmax, ymax = bboxes.split(1, dim=1)
			diff_width = torch.square((xmax - xmin) - AVG_WIDTH)
			diff_height = torch.square((ymax - ymin) - AVG_HEIGHT)
			diff = diff_width + diff_height
			sorted_boxes = torch.argsort(diff)[0]

		return feat[sorted_boxes], bboxes[sorted_boxes]

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
		output, im_scale = detectron.run_detectron(img_path)
		
		for qa in dataset[img_id]:
			points = qa['points']
			for ind, pt in enumerate(points):
				fname = './regionfeats_size/' + str(img_id) + ',' + str(pt['x']) + ',' + str(pt['y']) + '.pt'

				bbox_fname = './bboxes_size/' + str(img_id) + ',' + str(pt['x']) + ',' + str(pt['y']) + '.pt'
				
				img_feat, img_boxes = detectron._process_bbox(output, point=pt, points=None, feat_name='fc6', im_scale=im_scale)
				
				torch.save(img_feat, fname)
				torch.save(img_boxes, bbox_fname)
