import os
import re
import json

# pylint: disable=too-many-branches


def get_old_dic():
    ''' hacky get hardcoded reference dic
    '''
    cache = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)),
        '..',
        'cache')
    fname = os.path.join(cache, 'id_reference.json')
    with open(fname) as f:
        old_dic = json.load(f)
    return old_dic


def get_v3_to_v4_annos_dic(details):
    old_dic = get_old_dic()
    all_classes = set([int(r) for r in old_dic])

    new_partname2id = {}
    for cat in all_classes:
        new_partname2id[cat] = {
            'name': old_dic[str(cat)]['name'], 'parts': {}}
        for part in details.getParts(cat=cat):
            new_partname2id[cat]['parts'][part['name']] = part['part_id']

    v3_to_v4_convert = {}
    for cat in old_dic:
        cat_id = int(cat)
        v3_to_v4_convert[cat_id] = {}
        parts = old_dic[cat]['parts']
        choices = new_partname2id[cat_id]['parts']
        for part in parts:
            pname = parts[part]
            pname = pname.replace("_", "")
            pname = re.sub(r"\d", "", pname)
            # For train
            pname = re.sub("([hc])(left|right)", r'\1leftright', pname)
            if pname in choices:
                v3_to_v4_convert[cat_id][part] = choices[pname]
                continue

            # lower/upper legs merged in some (if not already matched)
            pname = re.sub("([lr])([fb])([lu])", r'\1\2', pname)
            # Hoofs and paws combined
            pname = re.sub("([lr])([fb])([ho]|[pa]){2}", r'\1\2hopa', pname)
            if pname in choices:
                v3_to_v4_convert[cat_id][part] = choices[pname]
                continue

            if pname == 'beak':
                if 'muzzle' in choices:
                    v3_to_v4_convert[cat_id][part] = choices['muzzle']
                    continue
                if 'nose' in choices:
                    v3_to_v4_convert[cat_id][part] = choices['nose']
                    continue
            elif pname == 'muzzle':
                if 'nose' in choices:
                    v3_to_v4_convert[cat_id][part] = choices['nose']
                    continue
                if 'beak' in choices:
                    v3_to_v4_convert[cat_id][part] = choices['beak']
                    continue
            elif pname == 'nose':
                if 'muzzle' in choices:
                    v3_to_v4_convert[cat_id][part] = choices['muzzle']
                    continue
                if 'beak' in choices:
                    v3_to_v4_convert[cat_id][part] = choices['beak']
                    continue

            if cat_id == 25:
                # bird
                if pname in ['lhorn', 'rhorn', 'lear', 'rear']:
                    # there aren't any real "horns" on birds - say head
                    # hardcoded
                    v3_to_v4_convert[cat_id][part] = choices['head']
                    continue
                elif pname in ['nose', 'muzzle']:
                    v3_to_v4_convert[cat_id][part] = choices['beak']
                    continue

            elif cat_id == 113:
                # hrose
                if pname == 'beak':
                    v3_to_v4_convert[cat_id][part] = choices['muzzle']
                    continue
                elif pname == 'lhorn':
                    v3_to_v4_convert[cat_id][part] = choices['lear']
                    continue
                elif pname == 'rhorn':
                    v3_to_v4_convert[cat_id][part] = choices['rear']
                    continue
            elif cat_id in [23, 258]:
                # Bike and motorbike
                if pname == 'body':
                    v3_to_v4_convert[cat_id][part] = choices['silh']
                    continue
                if pname == 'chainwheel':
                    v3_to_v4_convert[cat_id][part] = choices['silh']
                    continue
            elif cat_id == 427:
                if pname == 'screen':
                    v3_to_v4_convert[cat_id][part] = choices['framescreen']
                    continue
            elif cat_id == 65:
                if 'horn' in pname:
                    v3_to_v4_convert[cat_id][part] = choices[pname[0] + 'ear']
                    continue
    return v3_to_v4_convert
