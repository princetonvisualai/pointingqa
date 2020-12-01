'''
This module contains files which assist in visualizing pointwise results.
'''

import numpy as np
from skimage.morphology import medial_axis
import matplotlib.pyplot as plt


def get_best_representative(anns, obj_hist, obj, details, component='obj'):
    '''Selects the image which has a best (or reasonable) number and size of
    component parts for display purposes'''
    max_score = 0
    max_annotated = 0
    max_imid = 0
    max_size = 0
    for ann in anns:
        ids = []
        for part in ann['parts']:
            if part['part_id'] != 255:
                ids.append(part['part_id'])
        annotated = len(ids)
        score = 0
        for pid in ids:
            score += obj_hist[pid][component]
        if score > max_score:
            max_score = score
            max_imid = ann['image_id']
            max_annotated = annotated
            max_size = (details.getMask(img=ann['image_id'],
                                        cat=int(obj)) > 0).sum()
        elif score == max_score:
            mask = details.getMask(img=ann['image_id'], cat=int(obj))
            size = (mask > 0).sum()
            if annotated > max_annotated:
                max_score = score
                max_imid = ann['image_id']
                max_annotated = annotated
                max_size = size
            elif annotated == max_annotated:
                if size > max_size:
                    max_score = score
                    max_imid = ann['image_id']
                    max_annotated = annotated
                    max_size = size

    return max_imid


def gen_heatmap(hist, obj, details, show=False, component='obj'):
    '''Generate a heatmap representation of an object depicting the
    relative frequency of each part refering to the whole object'''
    anns = details.getAnns(cats=int(obj))
    max_imid = get_best_representative(anns, hist[obj], obj, details,
                                       component)
#     imp_cats = set(hist[obj].keys())
#     max_inter = 0
#     max_imid =0
#     for ann in anns:
#         parts = ann['parts']
#         imid = ann['image_id']
#         ids = set()
#         for part in parts:
#             ids.add(part['part_id'])
#         intersect = len(ids & imp_cats)
#         if intersect > max_inter:
#             max_inter = intersect
#             max_imid = imid
    mask = details.getMask(img=max_imid, cat=int(obj))
    unique = np.unique(mask)
    inst = int([unique[i] for i in range(len(unique)) if unique[i] != 0][0])
    mask = details.getMask(img=max_imid, cat=int(obj), instance=inst)
    heatmap = np.zeros(mask.shape)
    heatmap[mask == 100] = 2.0
    maxval = max([value[component] for value in hist[obj].values()])
    for k, value in hist[obj].items():
        value = value[component]
        inds = mask == int(k)
        if inds.sum() == 0:
            continue
        med, distance = medial_axis(inds, return_distance=True)
        distance[inds] /= distance[inds].max()
        distance[inds] *= 20.0
        heatmap[inds] += (float(value) / maxval) * 230
        heatmap[inds] += distance[inds]
    if show:
        plt.imshow(heatmap, cmap='hot')
        plt.show()
    return max_imid, heatmap
