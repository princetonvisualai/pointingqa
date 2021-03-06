from pythia.common.registry import registry
from pythia.tasks.base_dataset_builder import BaseDatasetBuilder
from pythia.tasks.vqa.objpart.dataset import ObjPartDataset

import json
import random
import copy
import numpy as np

@registry.register_builder("objpart")
class ObjPartBuilder(BaseDatasetBuilder):

	def __init__(self):
		super().__init__("objpart")
		self.writer = registry.get("writer")

		self.dataset_class = ObjPartDataset

	# Divides dataset into train, val, and test
	def get_dataset_div(self, vqamb_img, dataset_type, config):
		imgs_length = len(vqamb_img)
		train_cutoff = int(round(0.7 * imgs_length))
		val_cutoff = train_cutoff + int(round(0.1 * imgs_length))
		test_cutoff = val_cutoff + int(round(0.1 * imgs_length))
		keys = list(vqamb_img.keys())
		random.Random(4).shuffle(keys)
		
		train_set, val_set, test_set, final_set = [], [], [], []

		train_keys, val_keys, test_keys, final_keys = [], [], [], []
		for cnt, key in enumerate(keys):
			if cnt <= train_cutoff:
				train_keys.append(key)
				train_set.extend(vqamb_img[key])
			elif cnt <= val_cutoff:
				val_keys.append(key)
				val_set.extend(vqamb_img[key])
			elif cnt <= test_cutoff:
				test_keys.append(key)
				test_set.extend(vqamb_img[key])
			else:
				final_keys.append(key)
				final_set.extend(vqamb_img[key])

		# filter train, val, test if desired
		# train_set, val_set, test_set = self.filter_set(train_set), self.filter_set(val_set), self.filter_set(test_set)
		# final_set = self.filter_set(final_set)
		# print(len(test_set))
		div = {'train': train_set, 'val': val_set, 'test': test_set, 'final': final_set} # Change 'val'->val set
		# Train on unrolled, point-supervised training set
		# div['train'] = self.flatten_pt(div['train'])

		# Evalute on referred object task: separate example for each point
		# div['val'] = self.flatten_pt(div['val'])
		# div['test'] = self.flatten_pt(div['test'])

		# div['final'] = self.flatten_pt(div['final'])

		return div
					
	# Filter to only action questions 
	def filter_set(self, pt_dataset):
		filtered_set = []
		for qa in pt_dataset:
			if 'amb' not in qa and False: continue
			filtered_set.append(qa)
			
		return filtered_set
			
		# Intended for downloading files and actually "building" the dataset
	def _build(self, dataset_type, config):

		self.img_folder = ''
		
		pt_detectron_folder = ''
		pt_detectron_folder += config.pt_feat_folder + '/'

		self.detectron_folder = pt_detectron_folder
			
		objpart_path = ''
		with open(objpart_path, 'r') as f:
			objpart_img = json.load(f)
			
		self.div = self.get_dataset_div(objpart_img, dataset_type, config)

	def _load(self, dataset_type, config, *args, **kwargs):
		self.dataset = ObjPartDataset(
			dataset_type, config, detectron_folder=self.detectron_folder,
			objpart_div=self.div
		)

		return self.dataset

	def update_registry_for_model(self, config):
		# Register vocab (question and answer) sizes to registry for easy access to models.
		registry.register(
			self.dataset_name + "_text_vocab_size",
			self.dataset.text_processor.get_vocab_size(),
		)
		
		registry.register(
			self.dataset_name + "_num_final_outputs",
			self.dataset.answer_processor.get_vocab_size()-1,
		)
