# Computes region features for dataset (bounding boxes containing point)

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
	def _process_all_bbox(self, output, point, points, im_scale, feat_name):
		bboxes, scores, feat = output[0]['proposals'][0].bbox, output[0]['scores'], output[0][feat_name]

		bboxes = bboxes / im_scale

		scores = F.softmax(scores, dim=1)

		xmin, ymin, xmax, ymax = bboxes.split(1, dim=1)
		filt_pt = ((point['x'] >= xmin) & (point['x'] <= xmax) & (point['y'] >= ymin) & (point['y'] <= ymax)).flatten()
	    
		# Remove bboxes that contain other points (usually set to None)
		if points != None:
			for pt in points:
				if pt['ans'] != point['ans']:
					filt_ignore = ((pt['x'] >= xmin) & (pt['x'] <= xmax) & (pt['y'] >= ymin) & (pt['y'] <= ymax)).flatten()
					filt_pt = filt_pt * (~filt_ignore)
	    
		bboxes, scores, feat = bboxes[filt_pt], scores[filt_pt], feat[filt_pt]
		
		# check if no bounding boxes contain that point
		if bboxes.nelement() == 0:
			return torch.zeros(100, 2048), torch.zeros(100, 4)

		# sort bboxes by confidence (not performing NMS)
		max_scores, _ = scores[:, 1:].max(dim=-1)
		sorted_boxes = torch.argsort(max_scores, descending=True)[:100] # take maximum of 100 bounding boxes
		# This produces variable-length features - need to pad in data loader
		return feat[sorted_boxes], bboxes[sorted_boxes]
	
	def _process_nms_bbox(self, output, point, im_scale, feat_name):
		bboxes, scores, feat = output[0]['proposals'][0].bbox, output[0]['scores'], output[0][feat_name]

		bboxes = bboxes / im_scale

		scores = F.softmax(scores, dim=1)

		xmin, ymin, xmax, ymax = bboxes.split(1, dim=1)
		filt_pt = ((point['x'] >= xmin) & (point['x'] <= xmax) & (point['y'] >= ymin) & (point['y'] <= ymax)).flatten()

		bboxes, scores, feat = bboxes[filt_pt], scores[filt_pt], feat[filt_pt]

		# check if no bounding boxes contain that point																	    
		if bboxes.nelement() == 0:
			return torch.zeros(100, 2048), torch.zeros(100, 4)

		# perform nms
		max_conf = torch.zeros((scores.shape[0])).cuda()
		
		for cls_ind in range(1, scores.shape[1]):
			cls_scores = scores[:, cls_ind]
			keep = nms(bboxes, cls_scores, 0.5)
			max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
						     cls_scores[keep],
						     max_conf[keep])

		max_scores, _ = scores[:, 1:].max(dim=-1)
		filt_nms = (max_scores == max_conf)

		bboxes, scores, feat = bboxes[filt_nms], scores[filt_nms], feat[filt_nms]
		max_scores, _ = scores[:, 1:].max(dim=-1)
		sorted_boxes = torch.argsort(max_scores, descending=True)

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
	
	print(len(dataset))
	cnt = 0
	img_ids = list(dataset.keys())

	mode = 'post_bbox_all'

	for img_id in img_ids[args.lower:args.higher]:
		cnt = cnt + 1
		if cnt % 10 == 0:
			print(cnt)
		
		img_path = './images/' + str(img_id) + '.jpg'
		output, im_scale = detectron.run_detectron(img_path)
		
		# Precompute top N bounding boxes and their features.
		if 'topn' in mode:
			img_feat, keep_boxes = detectron._process_topn_bbox(output, point=None, feat_name='fc6', img_feat=None, keep_boxes=None)
		
		for qa in dataset[img_id]:
			points = [qa['point']]
			for ind, pt in enumerate(points):
				fname = './regionfeats/' + str(img_id) + ',' + str(pt['x']) + ',' + str(pt['y']) + '.pt'
				if os.path.exists(fname): continue

				if mode == 'post_bbox_all':
					img_feat, img_boxes = detectron._process_all_bbox(output, point=pt, points=None, feat_name='fc6', im_scale=im_scale)

				elif mode == 'post_bbox_nms':
					img_feat, img_boxes = detectron._process_nms_bbox(output, point=pt, feat_name='fc6', im_scale=im_scale)
				
				torch.save(img_feat, fname)

				bbox_fname = './bboxes/' + str(img_id) + ',' + str(pt['x']) + ',' + str(pt['y']) + '.pt'
				torch.save(img_boxes, bbox_fname)
