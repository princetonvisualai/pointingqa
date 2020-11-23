import json
import random
random.seed(4)

import torch
from torchvision import transforms

from pythia.common.registry import registry
from pythia.common.sample import Sample
from pythia.tasks.base_dataset import BaseDataset
from pythia.tasks.features_dataset import FeaturesDataset
from pythia.utils.text_utils import tokenize

import sys
# from detectron import PythiaDetectron

import copy
from PIL import Image

class VQAmbDataset(BaseDataset):

	def __init__(self, dataset_type, config, detectron_folder, bbox_folder, vqamb_div, *args, **kwargs):
		super().__init__("vqamb", dataset_type, config)

		self.dataset_type = dataset_type
		self.detectron_folder = detectron_folder
		self.bbox_folder = bbox_folder

		self.vqamb_data = vqamb_div[dataset_type]

		# self.detectron = PythiaDetectron()

		# hardcode these for now, can move them out later
		self.target_image_size = (448, 448)
		self.channel_mean = [0.485, 0.456, 0.406]
		self.channel_std = [0.229, 0.224, 0.225]

		self.cnt = 0
	
	def __len__(self):
		return len(self.vqamb_data)

	# Image transform for detectron - used if we don't precompute features
	def _image_transform(self, img):
		pass

	def get_item(self, idx):

		data = self.vqamb_data[idx]

		current_sample = Sample()

		# store queston and image id
		current_sample.img_id = data['id']
		current_sample.qa_id = data['qa_index']

		# store points
		current_sample.points = data['points']

		obj = data['all_objs'][0]
		xmin, ymin, xmax, ymax = obj['x'], obj['y'], obj['x'] + obj['w'], obj['y'] + obj['h']
		current_sample.gt_bbox = torch.Tensor([xmin, ymin, xmax, ymax])

		# process question
		question = data["question"]
		tokens = tokenize(question, remove=["?"])

		processed = self.text_processor({"tokens": tokens})
		current_sample.text = processed["text"]

		# process answers
		processed = self.answer_processor({"answers": data['all_ans']})
		current_sample.answers = processed["answers"]
		current_sample.targets = processed["answers_scores"][1:] # remove unknown index

		# Detectron features ----------------
		# TODO: read in detectron image instead if detectron is to be built
		detectron_path = self.detectron_folder + str(data['id'])
		bbox_path = self.bbox_folder + str(data['id'])
		if 'pt' in self.detectron_folder:
			point = data['points'][0]
			detectron_path += ',' + str(point['x']) + ',' + str(point['y'])
			bbox_path +=  ',' + str(point['x']) + ',' + str(point['y'])

		detectron_path += '.pt'
		bbox_path += '.pt'
		
		detectron_feat = torch.load(detectron_path, map_location=torch.device('cpu')).squeeze()
		# bbox_feat = torch.load(bbox_path, map_location=torch.device('cpu')).squeeze()
		'''if detectron_feat.shape[0] == 2048:
			detectron_feat = detectron_feat.unsqueeze(0)
			bbox_feat = bbox_feat.unsqueeze(0)
		'''
		'''
		if self.config.grid:
			 detectron_feat = detectron_feat.view(detectron_feat.shape[0], -1).T
		'''
		# x_down = max(int(round(pt['x']/600)), 18)
		# y_down = int(round(pt['y']/800), 25)

		# preproessing for grid features only
		# detectron_feat = detectron_feat.view(detectron_feat.shape[0], -1).T

		# Pad features to fixed length
		if self.config.grid:
			MAX_FEAT = 608

		else:
			MAX_FEAT = 100

		if self.config.pad_detectron:
			if detectron_feat.shape[0] > MAX_FEAT:
				detectron_feat = detectron_feat[:MAX_FEAT]
				# bbox_feat = bbox_feat[:MAX_FEAT]
			elif detectron_feat.shape[0] < MAX_FEAT:
				pad = torch.zeros(MAX_FEAT - detectron_feat.shape[0], detectron_feat.shape[1])
				detectron_feat = torch.cat([detectron_feat, pad], dim=0)
				pad = torch.zeros(MAX_FEAT - bbox_feat.shape[0], bbox_feat.shape[1])
				bbox_feat = torch.cat([bbox_feat, pad], dim=0)

		'''
		else:
			if detectron_feat.dim() > 1:
				detectron_feat = torch.zeros(2048)
		'''
		# current_sample.bbox = bbox_feat
		current_sample.image_feature_0 = detectron_feat
		# ---------------------------------------------

		return current_sample
