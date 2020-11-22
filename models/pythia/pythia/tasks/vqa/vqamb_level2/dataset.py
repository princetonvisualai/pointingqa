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

class VQAmb_Level2Dataset(BaseDataset):

	def __init__(self, dataset_type, config, img_folder, detectron_folder, 
		     context_folder, vqamb_div, *args, **kwargs):
		super().__init__("vqamb_level2", dataset_type, config)

		self.dataset_type = dataset_type
		self.img_folder = img_folder
		self.detectron_folder = detectron_folder
		self.context_folder = context_folder
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
		# current_sample.qa_id = data['qa_id']

		# store points
		current_sample.point = data['point'] # data['points']
		bbox = data['bbox']
		current_sample.gt_bbox = torch.Tensor([bbox['x'], bbox['y'], bbox['x'] + bbox['w'], bbox['y'] + bbox['h']])

		# process question
		question = data["pt_question"]
		tokens = tokenize(question, remove=["?"], keep=["'s"])

		processed = self.text_processor({"tokens": tokens})
		current_sample.text = processed["text"]

		# process answers
		processed = self.answer_processor({"answers": [data['ans']]})
		current_sample.answers = processed["answers"]
		current_sample.targets = processed["answers_scores"][1:] # remove unknown index

		# Detectron features ----------------
		# TODO: read in detectron image instead if detectron is to be built
		detectron_path = self.detectron_folder + str(data['id'])
		point = data['point'] # point = data['points'][0]
		if 'pt' in self.detectron_folder:
			detectron_path += ',' + str(point['x']) + ',' + str(point['y'])
		detectron_path += '.pt'
		
		detectron_feat = torch.load(detectron_path, map_location=torch.device('cpu'))

		# Pad features to fixed length
		if self.config.pad_detectron:
			if detectron_feat.shape[0] > 100:
				detectron_feat = detectron_feat[:100]
			elif detectron_feat.shape[0] < 100:
				pad = torch.zeros(100 - detectron_feat.shape[0], detectron_feat.shape[1])
				detectron_feat = torch.cat([detectron_feat, pad], dim=0)

		current_sample.image_feature_0 = detectron_feat
		# ---------------------------------------------

		# read in bounding boxes (hardcoded for now)
		
		bbox_path = ''
		bbox_path  += str(data['id']) + ',' + str(point['x']) + ',' + str(point['y']) + '.pt'
		bboxes = torch.load(bbox_path, map_location=torch.device('cpu'))

		if bboxes.shape[0] > 100:
			bboxes = bboxes[:100]
		elif bboxes.shape[0] < 100:
			pad = torch.zeros(100 - bboxes.shape[0], bboxes.shape[1])
			bboxes = torch.cat([bboxes, pad], dim=0)

		current_sample.pt_bbox = bboxes

		# read in image bounding boxes
		bbox_path = ''
		bbox_path  += str(data['id']) + '.pt' # + ',' + str(point['x']) + ',' + str(point['y']) + '.pt'
		bboxes = torch.load(bbox_path, map_location=torch.device('cpu'))

		if bboxes.shape[0] > 100:
			bboxes = bboxes[:100]
		elif bboxes.shape[0] < 100:
			pad = torch.zeros(100 - bboxes.shape[0], bboxes.shape[1])
			bboxes = torch.cat([bboxes, pad], dim=0)

		current_sample.img_bbox = bboxes
		
		# Context features --------------------
		if self.config.use_context:
			context_path = self.context_folder + str(data['id'])
			context_path += ',' + str(point['x']) + ',' + str(point['y'])
			context_path += '.pt'

			context_feat = torch.load(context_path, map_location=torch.device('cpu'))
			context_feat = context_feat.squeeze()
			orig_dim = context_feat.shape[0]

			if self.config.pad_context:
				if context_feat.shape[0] > 100:
					context_feat = context_feat[:100]
				elif context_feat.shape[0] < 100:
					pad = torch.zeros(100 - context_feat.shape[0], context_feat.shape[1])
					context_feat = torch.cat([context_feat, pad], dim=0)

			current_sample.context_feature_0 = context_feat
		# ---------------------------------------------

		return current_sample
