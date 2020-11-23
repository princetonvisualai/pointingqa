from visual_genome import api
import json
from collections import Counter
from collections import defaultdict
from random import random, shuffle, choices, randint
import pickle

# Use properties to generate a constrast map
# Attribute -> Contrasting attributes
def contrast_mapping(property_att_map):
	
	contrast_map = defaultdict(lambda: [])
	
	# Approach: assume all attributes in a property are contrasting, generate a contrast map accordingly
	for prop_atts in property_att_map.values():
		for i, att in enumerate(prop_atts):
			contrast_map[att] = prop_atts[0:i] + prop_atts[i+1:]
	
	return contrast_map

# Generate property -> attribute map
# Attribute -> Property
def property_mapping(property_att_map):
	
	reverse_map = defaultdict(lambda: "")
	
	for key in property_att_map:
		for att in property_att_map[key]:
			reverse_map[att] = key
		
	return reverse_map

# Given two objects, generate all possible questions in which the two objects
# contrast in the attribute in question
def find_contrast(obj1, obj2, constrast_map, category_map):
	att1 = [x.strip().lower() for x in obj1['attributes']]
	att2 = [x.strip().lower() for x in obj2['attributes']]
	
	qas = []
	
	for i in range(len(att1)):
		for j in range(len(att2)):
			a = att1[i]
			b = att2[j]
			# need to make sure the entities are genuinely contrasting
			if((b in contrast_map[a] or a in contrast_map[b]) and (a not in att2) and (b not in att1)):
				# get general category from either a or b
				if b in category_map.keys():
					# construct the question
					question = "What " + category_map[b] + " is the " + obj1['names'][0]
					if category_map[b] == 'action': question += " doing"
					question += "?"
					
					# randomly select either object attribute to be the correct answer
					if random() > 0.5:
						answer = a
						qas.append((question, answer, obj1))
					else:
						answer = b
						qas.append((question, answer, obj2))
			
	return qas

# Calculate PDF of 2D locations of points
def convert_to_pdf(locs, rnd_buckets):

	pdf = {}
	for (x_loc, y_loc) in locs:
		x_rnd = round(x_loc, 2)
		y_rnd = round(y_loc, 2)
		pdf[(x_rnd, y_rnd)] = pdf.get((x_rnd, y_rnd), 0) + 1

	return pdf

# Read in attributes data
print("Loading attributes data...")
filename = './attributes.json'
with open(filename, 'r') as f:
	att_data = json.load(f)

# Collect a count of all the attributes
# unique_att = {}
# for i in range(len(att_data)):
# 	for objects in att_data[i]['attributes']:
# 		atts = objects.get('attributes')
# 		if atts == None: continue
# 		for att in atts:
# 			att = att.strip().lower()
# 			unique_att.update({att : unique_att.get(att, 1) + 1})

# Define property map for top 100 attributes
property_att_map = {
	'color': ['white', 'black', 'blue', 'green', 'red', 'brown', 'yellow', 'gray', 'silver', 'orange', 
		  'grey', 'pink', 'tan', 'purple', 'gold', 'beige', 'blonde', 'light blue', 'light brown'], # 'plaid',
	'shape': ['round', 'rectangular', 'square'],
	'action': ['standing', 'sitting', 'walking', 'hanging', 'playing', 'looking', 'watching']
}

collapse_map = {
	'light brown': 'brown',
	'beige': 'brown',
	'tan': 'brown',
	'light blue': 'blue',
	'blonde': 'yellow',
	'gold': 'yellow',
	'watching': 'looking',
	'silver': 'gray',
	'circular': 'round',
	'grey': 'gray'
}

collapse_keys = list(collapse_map.keys())

contrast_map = contrast_mapping(property_att_map)
property_map = property_mapping(property_att_map)

print("Loading point pdf data...")
# Read in normalized point locations from "What's the point" data (Bearman et al.)
with open ('./norm_locs', 'rb') as fp:
	norm_locs = pickle.load(fp)

locs_pdf = convert_to_pdf(norm_locs, 2)

print("Creating VQAmb Dataset...")
# (img id, question, answer, possible answers, all objects, point)
QA_dataset = {}

for img in range(len(att_data)):

	if img % 100 == 0: print(img)

	obj_to_att = defaultdict(lambda: [])

	objects = att_data[img]['attributes']
	shuffle(objects)
	
	# find all objects in image (used to find objects with multiple occurrence)
	for index, obj in enumerate(objects):
		name = obj['names'][0]
		atts = obj.get('attributes')
		if atts == None: continue
		obj_to_att[name].append(index)

	for obj_name in obj_to_att:
		instances = obj_to_att[obj_name]
		# if not (len(instances) >= 2 and len(instances) <= 4): continue 

		objs = [att_data[img]['attributes'][x] for x in instances]

		# Iterate through each property, see if question can be asked
		for prop in property_att_map:
			prop_answers = []
			prop_objs = []
			for obj in objs:
				# collect all attributes under the property for this object
				atts = [x.strip().lower() for x in obj['attributes']]
				prop_atts = [att for att in atts if att in property_att_map[prop]]
				prop_atts = [collapse_map[att] if att in collapse_keys else att for att in prop_atts] # add grey/gray exception
				if len(set(prop_atts)) == 0 or len(set(prop_atts)) >= 2: continue

				prop_answers.append(list(set(prop_atts)))
				prop_objs.append(obj)

			# neat way of checking that there are at least two objects with disjoint answers
			if len(prop_answers) <= 1: continue
			if len(set.intersection(*map(set, prop_answers))) > 0: continue

			img_id = att_data[img]['image_id']

			# Construct question
			question = "What " + prop + " is the " + obj_name
			if prop == 'action': 
				question += " doing"
			question += "?"

			# Select answer, referred object
			# ans, obj_ans = prop_answers[ans_ind], prop_objs[ans_ind]

			# loc = choices(list(locs_pdf.keys()), list(locs_pdf.values()))[0]
			# x_coor = int(round(obj_ans['x'] + loc[0] * obj_ans['w']))	
			# y_coor = int(round(obj_ans['y'] + loc[1] * obj_ans['h']))
			# point = (x_coor, y_coor)

			# Generate points for dataset
			points = []
			for ans, obj in zip(prop_answers, prop_objs):
				# loc = choices(list(locs_pdf.keys()), list(locs_pdf.values()))[0] # sample location in bounding box
				loc = [0.5, 0.5]
				x_coor = int(round(obj['x'] + loc[0] * obj['w']))	
				y_coor = int(round(obj['y'] + loc[1] * obj['h']))

				point = {'x': x_coor, 'y': y_coor, 'ans': ans[0]} # currently one point per relevant object
				points.append(point)

			# flatten answers into one list
			prop_answers = list(set(sum(prop_answers, [])))
			print(prop_objs)
			break
			# TODO: correspond each answer with each object, so we can generate grounding masks for supervision
			qa_entry = {'id': img_id, 'question': question, 'all_ans': prop_answers, 'all_objs': prop_objs, 'points': points}
			curr_list = QA_dataset.get(img_id, [])
			curr_list.append(qa_entry)
			QA_dataset.update({img_id : curr_list})
	

# for img in range(len(att_data)):

# 	print(img)

# 	obj_to_att = defaultdict(lambda: [])

# 	objects = att_data[img]['attributes']
# 	shuffle(objects)
	
# 	# find all objects in image (used to find objects with multiple occurrence)
# 	for index, obj in enumerate(objects):
# 		name = obj['names'][0]
# 		atts = obj.get('attributes')
# 		if atts == None: continue
# 		obj_to_att[name].append(index)

# 	# Iterate over types of objects in the image
# 	for obj in obj_to_att:
# 		# only consider objects with >=2 instances
# 		instances = obj_to_att[obj]
# 		if len(instances) == 1: continue
		
# 		# All objects of the type
# 		objs = [att_data[img]['attributes'][x] for x in instances]

# 		# consider all pairs of objects with >=2 instances, find contrasting pairs
# 		unique_qs = set() # make sure we don't repeat a question for an image
# 		for i in range(len(instances)):
# 			for j in range(i+1, len(instances)):
# 				obji = att_data[img]['attributes'][instances[i]]
# 				objj = att_data[img]['attributes'][instances[j]]

# 				# Check if there is a contrast, return the appropriate question if so
# 				qas = find_contrast(obji, objj, contrast_map, property_map)

# 				for (question, answer, obj_corr) in qas:
# 					if question in unique_qs: continue

# 					# Sample a normalized bbox point and convert it to a point in the image
# 					loc = choices(list(locs_pdf.keys()), list(locs_pdf.values()))[0]
# 					x_coor = int(round(obj_corr['x'] + loc[0] * obj_corr['w']))	
# 					y_coor = int(round(obj_corr['y'] + loc[1] * obj_corr['h']))
# 					point = (x_coor, y_coor)

# 					img_id = att_data[img]['image_id']
# 					QA_dataset.append((img_id, question, answer, obj_corr, objs, point))
# 					unique_qs.add(question)

import json
with open('VQAmb_Dataset.json', 'w') as outfile:
    json.dump(QA_dataset, outfile)

# with open('qadataset_alpha2', 'wb') as fp:
# 	pickle.dump(QA_dataset, fp)

# with open('qadataset_alpha', 'rb') as fp:
# 	QA_dataset = pickle.load(fp)


