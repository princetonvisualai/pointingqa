import json
from scipy.stats import mode
from collections import Counter

CLASSES = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]

# skip = ['background', 'boat', 'diningtable', 'chair', 'sofa', 'tvmonitor']

# obj_cl = []
# part_cl = []
# for cl in CLASSES:
# 	if cl in skip: continue
# 	obj_cl.append(cl + 'object')
# 	part_cl.append(cl + 'part')

# obj_cl.extend(part_cl)

# print(obj_cl)

with open('pascal_gt.json', 'r') as f:
	a = json.load(f)

print(len(a))

pascal_gt_clean = {}

objpart_d1 = {}
cnt = 0
all_ans = []
for img in a:
	# print(img)
	qas = []
	pts = {}
	for pt in a[img]:
		qa = {}
		qa['id'] = img
		qa['question'] = "What is this referring to?"
		# qa['question'] = "Is this "
		num_answer = mode(a[img][pt])[0][0]

		# skip negative labels
		if num_answer < 0:
			continue

		z = Counter(a[img][pt]).most_common()

		if len(z) > 1 and (z[0][1] == z[1][1]):
			continue

		pts[pt] = a[img][pt]

		# mark if point involves human disagreement
		if len(set(a[img][pt])) > 1:
			qa['amb'] = True

		qa['question'] = "What is this referring to?"

		# qa['question'] += CLASSES[num_answer % 20] + " point referring to a part?"

		# print(qa['question'])

		if num_answer > 20:
			ans = "part"

		else:
			ans = "object"

		qa['ans'] = ans
		qa['object'] = CLASSES[num_answer % 20]

		# all_ans.append(ans)

		x, y = [int(coor) for coor in pt.split('_')]
		qa['point'] = {'ans': qa['ans'], 'x': x, 'y': y}

		qas.append(qa)
		cnt += 1

	if qas:
		objpart_d1[img] = qas
		pascal_gt_clean[img] = pts

with open('objpart_d3new.json', 'w') as f:
	json.dump(objpart_d1, f)

# with open('pascal_gt_clean.json', 'w') as f:
# 	json.dump(pascal_gt_clean, f)
