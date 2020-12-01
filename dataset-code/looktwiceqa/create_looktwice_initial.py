# Format of question: How many of these are there?

import json
import random
random.seed(6)
import copy
import pickle
import inflect

def process_ans(a):
	a = a.strip('.').lower()
	words = a.split()

	word_dict = {'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7'}
	num_list = ['1', '2', '3', '4', '5', '6', '7']

	true_ans = '0'
	for word in words:
		if word in word_dict:
			true_ans = word_dict[word]
			break

		elif word in num_list:
			true_ans = word
			break

	return true_ans

with open('all_objs.pkl', 'rb') as f:
	all_objs = pickle.load(f)

p = inflect.engine()
plural_to_singular = {}
for key in all_objs:
	plural_to_singular[key] = p.singular_noun(key)

print({k: v for k, v in sorted(all_objs.items(), key=lambda item: item[1], reverse=True)}) 

with open('question_answers.json', 'r') as f:
	qas = json.load(f)

with open('objects.json', 'r') as f:
	atts = json.load(f)

cnt = 0
# QA pair with point, QA pair with image
howmany_dataset = {}
all_objs = {}
for qa_scene, att_scene in zip(qas, atts):
	all_qas = []
	for qa in qa_scene["qas"]:
		q, a = qa['question'], process_ans(qa['answer'])
		if "many" not in q or ("many" in q and a in ["none", "0", "zero"]):
			continue

		cnt += 1

		# fix questions without verbs
		words = q.split()
		if len(words) == 3:
			q = q.strip("?") + " are there?"
			words = q.split()

		# filter out questions with compound nouns
		if not words[3] in ["are", "is", "can", "does", "do", "in"] and "ing" not in words[3]:
			continue 

		if cnt % 1000 == 0:
			print(cnt)

		obj_type = words[2].strip('?')
		if obj_type == 'bikes': 
			obj_type = 'bicycles'
		elif obj_type == 'airplanes': 
			obj_type = 'planes'
		if p.singular_noun(obj_type) == False:
			obj_type = p.plural(obj_type)
			words[2] = obj_type
			q = ' '.join(words)

		all_objs.update({obj_type: all_objs.get(obj_type, 0) + 1})

		qa = {}
		qa['id'] = att_scene['image_id']

		qa['img_question'] = q
		qa['ans'] = a

		qa['qtype'] = 'howmany'

		relevant_obj = []
		for obj in att_scene['objects']:
			name = obj['names'][0] 
			if name == plural_to_singular[obj_type] or obj_type=='people' and name in ['man', 'woman', 'person']:
				relevant_obj.append(obj)

		if not relevant_obj: continue
		if obj_type == 'shirts': 
			print("yay")
		# cnt += 1

		choice = random.choice(relevant_obj)
		point = {'x': int(round((choice['x'] + choice['w']/2))), 
					'y': int(round((choice['y'] + choice['h']/2))), 'ans': a}
		qa['point'] = point
		qa['bbox'] = choice

		qa['pt_question'] = q.replace(obj_type, 'of these')

		qa['obj_type'] = obj_type
		all_qas.append(qa)

	if not all_qas:
		continue

	howmany_dataset.update({att_scene['image_id']: all_qas})

with open('all_objs.pkl', 'wb') as f:
	pickle.dump(all_objs, f)
