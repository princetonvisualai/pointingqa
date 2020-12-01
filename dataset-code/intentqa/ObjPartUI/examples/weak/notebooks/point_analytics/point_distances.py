'''This module provides functions to compute statistics
   based on centroids of part masks'''
import collections
import numpy as np
from skimage.measure import moments


__all__ = [
    "get_centroid",
    "collect_all_mask_centroids",
    "collect_distances_to_centroids",
    "add_centroid_distances"]


def get_centroid(mask, bRound=True):
    M = moments(mask.astype(np.uint8), 1)
    x, y = M[1, 0] / M[0, 0], M[0, 1] / M[0, 0]
    if bRound:
        return int(round(x)), int(round(y))
    return x, y


def collect_all_mask_centroids(details):
    imids = []
    for det in details.getCats():
        imids.extend(det['images'])
    imids = list(set(imids))  # Compute each once
    anns = details.getAnns(imgs=imids)
    all_centroids = collections.defaultdict(dict)
    for ann in anns:
        annid = ann['id']
        parts = ann['parts']
        imid = str(ann['image_id'])
        imid = "{}_{}".format(imid[0:4], imid[4:])
        all_centroids[imid][annid] = {}
        for part in parts:
            pid = int(part['part_id'])
            if pid in [
                    0,
                    255]:  # Don't compute center for "hard" or "background"
                continue
            mask = details.decodeMask(part['segmentation'])
            cc, cr = get_centroid(mask)
            all_centroids[imid][annid][pid] = (int(cc), int(cr))
    return all_centroids


def collect_distances_to_centroids(dInfo, details, cached):
    '''Instance IDs are currently stored as 0,1,etc.
        but have unique ids otherwise'''
    key2inst = {}
    for key, data in dInfo.items():
        if key in cached:
            key2inst[key] = cached[key]
            continue
        imid = data['imid']
        imid = int("".join(imid.split("_")))
        obj = data['obj_id']
        # inst = int(data['inst_id'])
        mask = details.getMask(img=imid, cat=int(obj))
        i, j = int(data['yCoord']), int(data['xCoord'])
        key2inst[key] = int(mask[i, j])
    return key2inst


def get_euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def add_centroid_distances(dInfo, all_centroids, annid_by_point):
    copy = {k: v for k, v in dInfo.items()}
    for k, v in copy.items():
        annids = annid_by_point[k]
        imid = v['imid']
        part_centers = all_centroids[imid]
        pX, pY = int(v['xCoord']), int(v['yCoord'])
        all_distances = {}
        # part_id = v['part_id']
        for annid in annids:
            distances = {}
            for part, (x, y) in part_centers[str(annid)].items():
                dist = float(np.linalg.norm(pX - x) + np.linalg.norm(pY - y))
                distances[part] = dist
            all_distances[annid] = distances
        copy[k]['centroid_distances'] = all_distances
    return copy
