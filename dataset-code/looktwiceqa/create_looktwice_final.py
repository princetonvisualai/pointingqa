import json
import random
import inflect

def compute_iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
	boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

	# compute the area of intersection rectangle
	interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
	# compute the area of both the prediction and ground-truth
	# rectangles

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	if boxAArea == 0:
		return 0, 0, 0

	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou, boxAArea, boxBArea

def iou_filter(all_people, thresh=0.25):

	eliminate = []
	for i in range(len(all_people)):
		personi = all_people[i]
		bboxi = (personi['x'], personi['y'], personi['x']+personi['w'], personi['y']+personi['h'])
		for j in range(i+1, len(all_people)):
			personj = all_people[j]
			bboxj = (personj['x'], personj['y'], personj['x']+personj['w'], personj['y']+personj['h'])
			iou, iarea, jarea = compute_iou(bboxi, bboxj)
			if iou > thresh:
				rmv = i if iarea > jarea else j
				eliminate.append(rmv)

	filter_people = []
	for i in range(len(all_people)):
		if i not in eliminate:
			filter_people.append(all_people[i])
	return filter_people

with open('', 'r') as f:
	howmany = json.load(f)

# First step: filter objects by count or if in a list of removed objects -----
obj_cnts = {}

for img_id in howmany:
	for qa in howmany[img_id]:
		obj_cnts.update({qa['obj_type']: obj_cnts.get(qa['obj_type'], 0) + 1})

# print(obj_cnts['shirts'])
# print({k: v for k, v in sorted(obj_cnts.items(), key=lambda item: item[1], reverse=True)})

rmv_obj = ['men', 'women', 'pizzas', 'wheels', 'beds', 'hands', 'players', 'buildings',
			'boys', 'legs', 'ears', 'animals', 'eyes', 'girls', 'tables',
			'wings', 'children', 'tires', 'shoes']

p = inflect.engine()
keep_obj = []
for obj in obj_cnts:
	if obj_cnts[obj] >= 100 and obj not in rmv_obj:
		keep_obj.append(p.singular_noun(obj))

keep_obj.remove("light")
keep_obj.remove("window")
keep_obj.remove("tree")

animals = ['people', 'dogs', 'cats', 'giraffes', 'horses', 'zebras', 'elephants', 'birds', 'bears', 'cows', 'sheep', 'animals']
vehicles = ['trains', 'buses', 'cars', 'planes', 'boats', 'motorcycles', 'trucks', 'skateboards', 'surfboards', 'bicycles']
# keep_obj = animals + vehicles

howmany_filter = {}

filtered_obj = []

cnt = 0
person_cnt = 0
for img_id in howmany:
	for qa in howmany[img_id]:
		if obj_cnts[qa['obj_type']] < 100 or qa['obj_type'] in rmv_obj:
			continue

		# handle animals and vehicles
		if qa['obj_type'] in animals:
			words = qa['pt_question'].split()
			words.insert(4, "beings")
			qa['pt_question'] = ' '.join(words)

		elif qa['obj_type'] in vehicles:
			words = qa['pt_question'].split()
			words.insert(4, "vehicles")
			qa['pt_question'] = ' '.join(words)

		else:
			words = qa['pt_question'].split()
			words.insert(4, "objects")
			qa['pt_question'] = ' '.join(words)

		filtered_obj.append(qa['obj_type'])
		cnt += 1
		howmany_filter.update({qa['id']: howmany_filter.get(qa['id'], []) + [qa]})
# ------------------------------------------------------------------------------

# Filter to only keep images with two unique answers involving two unique objects -----
howmany_train = {}
howmany_final_test = {}
cnt = 0

qa_noadd = []

for img_id in howmany_filter:
	if len(howmany_filter[img_id]) == 1:
		qa = howmany_filter[img_id][0]
		howmany_train.update({qa['id']: howmany_train.get(qa['id'], []) + [qa]})
		continue

	valid = False
	all_ans = []
	all_obj = []
	all_qtype = []
	for qa in howmany_filter[img_id]:
		if len(all_ans) == 0:
			all_ans.append(qa['ans'])
			all_obj.append(qa['obj_type'])
			all_qtype.append(qa['pt_question'].split()[4])

		else:
			if qa['ans'] != all_ans[0] and qa['obj_type'] != all_obj[0]:
				# if you want to enforce differences between categories
				# q_type = qa['pt_question'].split()[4]
				# if q_type == all_qtype[0]:
				valid = True

	if valid:
		for qa in howmany_filter[img_id]:
			howmany_final_test.update({qa['id']: howmany_final_test.get(qa['id'], []) + [qa]})

	else:
		for qa in howmany_filter[img_id]:
			howmany_train.update({qa['id']: howmany_train.get(qa['id'], []) + [qa]})

# divide this into val and test --------
keys = list(howmany_final_test.keys())
random.Random(4).shuffle(keys)

howmany_final_test2 = {}
cnt = 0
for img_id in keys:
	howmany_final_test2[img_id] = []
	cnt += len(howmany_final_test[img_id])
	for qa in howmany_final_test[img_id]:
		if cnt < 1000:
			qa['div'] = 'val'
		else:
			qa['div'] = 'test'

		howmany_final_test2[img_id].append(qa)

with open('', 'w') as outfile:
	json.dump(howmany_final_test2, outfile)

print("Written!")
# ---------------------------------------------------------------------------------------

# For images with one question, generate a "how many" question w/ a different answer using VG annotations ---

# Get singular versions of object types
p = inflect.engine()
plural_to_singular = {}
for key in filtered_obj:
	plural_to_singular[key] = p.singular_noun(key)

singular_to_plural = {v: k for k, v in plural_to_singular.items()}

filtered_objs_singular = list(plural_to_singular.values())

with open('objects.json', 'r') as f:
	vg = json.load(f)
print("Loaded!")

cnt = 0
img_ids = set(list(howmany_train.keys()))
howmany_final_train = {}
# rmv_obj = ['light', 'window', 'tree']

add_qas = []
for scene in vg:
	img_id = scene['image_id']

	# if img_id != 2401914: continue

	if img_id not in img_ids:
		continue 

	qas = howmany_train[img_id]
	human_objects = [plural_to_singular[qa['obj_type']] for qa in qas]

	# form object map
	obj_map = {}
	# assemble obj -> cnt dict
	for obj in scene['objects']:
		if not obj['synsets']:
			name = obj['names'][0]
		else:
			name = obj['synsets'][0].split('.')[0]
		if name in ['man', 'woman', 'boy', 'girl', 'player', 'child']:
			name = 'person'
		if name not in filtered_objs_singular: continue
		if (name not in human_objects):
				# or (orig_obj_type=='people' and name not in ['person', 'man', 'woman'])):
			obj_map.update({name: obj_map.get(name, []) + [obj]})

	# IoU filtering
	for obj in obj_map:
		iou_obj = iou_filter(obj_map[obj], thresh=0.2)
		obj_map[obj] = iou_obj

	used = []

	for qa in qas:

		orig_ans = qa['ans']

		ans_candidates = [x for x in list(obj_map.keys()) if len(obj_map[x]) != int(orig_ans) and len(obj_map[x]) <= 7]
		ans_candidates = [x for x in ans_candidates if x in keep_obj and x not in used]
		if not ans_candidates:
			qa['div'] = 'train'
			add_qas.append(qa)
			continue
			# howmany_final_train.update({qa['id']: howmany_final_train.get(qa['id'], []) + [qa]})
			# continue

		# 	qa_noadd.append(qa)
		# # 	qa['div'] = 'train'
		# # 	howmany_final_train[img_id] = [qa]
		# # 	cnt += 1
		# 	continue

		choice = random.Random(4).choice(ans_candidates)
		used.append(choice)
		choice_plural = singular_to_plural[choice]

		qa_add = {}
		qa_add['id'] = img_id
		qa_add['img_question'] = "How many " + choice_plural + " are there?"

		if choice_plural in animals:
			category = "beings"
		elif choice_plural in vehicles:
			category = "vehicles"
		else:
			category = "objects"

		qa_add['pt_question'] = "How many of these " + category + " are there?"
		qa_add['ans'] = str(len(obj_map[choice]))
		bbox = random.Random(4).choice(obj_map[choice])
		qa_add['point'] = {'x': int(round((bbox['x'] + bbox['w']/2))), 
							'y': int(round((bbox['y'] + bbox['h']/2))), 'ans': qa_add['ans']}
		qa_add['bbox'] = bbox
		qa_add['div'] = 'trainadd'
		qa_add['obj_type'] = singular_to_plural[choice]

		qa['div'] = 'train'

		howmany_final_train.update({qa['id']: howmany_final_train.get(qa['id'], []) + [qa, qa_add]})
		cnt += 2

# --------------------------------------------------------------------------------------

# add more examples to training set -----

# add all remaining examples
# for qa in qa_noadd:
# 	howmany_final_train.update({qa['id']: howmany_final_train.get(qa['id'], []) + [qa]})

random.Random(4).shuffle(add_qas)
not_one_cnt = 0
for qa in add_qas:
	if qa['ans'] != '1':
		qa['div'] = 'train'
		not_one_cnt += 1
		howmany_final_train.update({qa['id']: howmany_final_train.get(qa['id'], []) + [qa]})

one_cnt = 0
for qa in add_qas:
	if qa['ans'] != '1' or one_cnt > 0.5 * not_one_cnt:
		continue
	one_cnt += 1
	qa['div'] = 'train'
	howmany_final_train.update({qa['id']: howmany_final_train.get(qa['id'], []) + [qa]})
#----------------------------------------

with open('howmany_addans.json', 'w') as outfile:
	json.dump(howmany_final_train, outfile)

# Combine test and train ----------

howmany_final = {}

for img_id in howmany_final_train:
	howmany_final[img_id] = howmany_final_train[img_id]

for img_id in howmany_final_test2:
	howmany_final[img_id] = howmany_final_test2[img_id]

with open('howmany_balanceimgs_eq.json', 'w') as outfile:
	json.dump(howmany_final, outfile)

