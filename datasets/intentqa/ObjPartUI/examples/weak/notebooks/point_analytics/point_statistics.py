'''This module provides functions to compute statistics
based on centroids of part masks'''
import collections
import importlib
import skimage.measure
import numpy as np
import point_analytics
importlib.import_module('.utils', 'point_analytics')

__all__ = [
    "select_unambiguous_parts",
    "add_contrarian_response",
    "get_centroid",
    "collect_all_mask_centroids",
    "collect_distances_to_centroids",
    "add_centroid_distances"]


def select_unambiguous_parts(part_hist, threshold=0.9, avoid_perfect=False):
    '''Selects the parts of each object which tend to be ambiguous, where
       certainty is measured as having the frequency of responses being
       above a predefined threshold. Returns a dictionary of parts
       which tend to be unambiguous, with the values of each entry
       being a tuple of the frequency of choosing the usual
        class and the less common response.
        '''
    dUnAmbiguous = {}
    for cls, parts in part_hist.items():
        dUnAmbiguous[cls] = {}
        for part, freqs in parts.items():
            if int(part) == 100:
                continue
            obj_freq = freqs['obj']
            part_freq = freqs['part']
            unambiguous_freq = obj_freq + part_freq
            if part_freq > threshold or obj_freq > threshold:
                if avoid_perfect and (part_freq == 1.0 or obj_freq == 1.0):
                    continue
                least_common = 'obj' if obj_freq < part_freq else 'part'
                dUnAmbiguous[cls][int(part)] = (unambiguous_freq, least_common)
#             ambiguous_freq = 1.0 - obj_freq - part_freq
#             if ambiguous_freq > threshold and ambiguous_freq != 1.0:
#                 most_common = 'obj' if obj_freq > part_freq else 'part'
#                 dUnAmbiguous[cls][int(part)] = (ambiguous_freq, most_common)
    return dUnAmbiguous


def add_contrarian_response(question, within_set, contrarian, common, dInfo):
    '''
    Honestly, I'm not sure what this does. It is mapped to each response and
    adds those generated from object mask with response of part (and vice
    versa).
    Deprecated.
    '''
    keys = list(question)
    for key in keys:
        key_data = point_analytics.utils.separate_key(key)
        obj = key_data['obj_id']
        part = dInfo[key]['nearest_part_id']
        if obj not in contrarian:
            contrarian[obj] = {}
            common[obj] = {}
        ans = question[key]['answer']
        if part not in within_set[obj]:
            if part not in common[obj]:
                common[obj][part] = []
            common[obj][part].append(question)
            continue

        _, component = within_set[obj][part]
        if int(ans) == 100:
            if component == 'obj':
                if part not in contrarian[obj]:
                    contrarian[obj][part] = []
                contrarian[obj][part].append(question)
                continue
        else:
            if component == 'part':
                if part not in contrarian[obj]:
                    contrarian[obj][part] = []
                contrarian[obj][part].append(question)
                continue
        if part not in common[obj]:
            common[obj][part] = []
        common[obj][part].append(question)


def get_centroid(mask, bRound=True):
    M = skimage.measure.moments(mask.astype(np.uint8), 1)
    x, y = M[1, 0] / M[0, 0], M[0, 1] / M[0, 0]
    if bRound:
        return int(round(x)), int(round(y))
    return x, y


def collect_all_mask_centroids(details, track=None):
    imids = []
    for det in details.getCats():
        imids.extend(det['images'])
    imids = list(set(imids))  # Compute each once
    anns = details.getAnns(imgs=imids)
    all_centroids = collections.defaultdict(dict)
    if not isinstance(track, type(None)):
        import tqdm
    if track == "notebook":
        tracker = tqdm.tqdm_notebook(anns)
    elif track == "script":
        tracker = tqdm.tqdm(anns)
    else:
        tracker = anns
    for ann in tracker:
        annid = ann['id']
        category_id = ann['category_id']
        parts = ann['parts']
        imid = str(ann['image_id'])
        imid = "{}_{}".format(imid[0:4], imid[4:])
        if category_id not in all_centroids[imid]:
            all_centroids[imid][category_id] = {}
        if annid not in all_centroids[imid][category_id]:
            all_centroids[imid][category_id][annid] = {}
        # Document instance level segmentation
        # Store in part "100" (replace phony silhouette)
        mask = details.decodeMask(ann['segmentation'])
        cc, cr = get_centroid(mask)  # centroid column, centroid row
        all_centroids[imid][category_id][annid][100] = (int(cc), int(cr))
        for part in parts:
            pid = int(part['part_id'])
            if pid in [
                    0,
                    100,   # Don't compute for the "silhouette" class
                    255]:  # Don't compute for "hard" or "background" classes
                continue
            mask = details.decodeMask(part['segmentation'])
            cc, cr = get_centroid(mask)
            all_centroids[imid][category_id][annid][pid] = (int(cc), int(cr))
    return dict(all_centroids)


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
