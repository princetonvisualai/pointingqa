import re


def editSides(references):
    conversions = {
        'lebrow': 'left eye brow',
        'rebrow': 'right eye brow',
        'lfpa': 'left front paw',
        'rfpa': 'right front paw',
        'lbpa': 'left back paw',
        'rbpa': 'right back paw',
        'llleg': 'left lower leg',
        'luleg': 'left upper leg',
        'rlleg': 'right lower leg',
        'ruleg': 'right upper leg',
        'luarm': 'left upper arm',
        'llarm': 'left lower arm',
        'ruarm': 'right upper arm',
        'rlarm': 'right lower arm',
        'lfleg': 'left front leg',
        'lflleg': 'left front lower leg',
        'lfuleg': 'left front upper leg',
        'lfho': 'left front hoof',
        'rfleg': 'right front leg',
        'rflleg': 'right front lower leg',
        'rfuleg': 'right front upper leg',
        'rfho': 'right front hoof',
        'lbleg': 'left back leg',
        'lblleg': 'left back lower leg',
        'lbuleg': 'left back upper leg',
        'lbho': 'left back hoof',
        'rbleg': 'right back leg',
        'rblleg': 'right back lower leg',
        'rbuleg': 'right back upper leg',
        'rbho': 'right back hoof'}

    reference_ = dict(references.copy())
    for k, v in reference_.items():
        #         if k in tooSmall:
        #             reference_.pop(k, None)
        #             continue
        parts = v['parts']
        for part_id, part_name in parts.items():
            if part_id == 100:
                reference_[k]['parts'].pop(part_id, None)
                continue
            if part_name in conversions:
                reference_[k]['parts'][part_id] = conversions[part_name]
            elif part_name[0] == 'lbl':
                reference_[
                    k]['parts'][part_id] = 'left bottom lower' + part_name[3:]
            elif part_name[0] == 'lbl':
                reference_[
                    k]['parts'][part_id] = 'left bottom lower' + part_name[3:]
            elif part_name[0] == 'l':
                reference_[k]['parts'][part_id] = 'left ' + part_name[1:]
            elif part_name[0] == 'r':
                reference_[k]['parts'][part_id] = 'right ' + part_name[1:]
            elif part_name[0] == 'f':
                reference_[k]['parts'][part_id] = 'front ' + part_name[1:]
            elif part_name[0:2] in ['bw', 'bl']:
                reference_[k]['parts'][part_id] = 'back ' + part_name[1:]
            elif part_name[0:2] in ['cr', 'cb', 'cl']:
                reference_[k]['parts'][part_id] = 'coach ' + part_name[1:]
            m = re.search('(?:_[0-9])$', reference_[k]['parts'][part_id])
#             print(reference_[k]['parts'][part_id], m)
            if m:
                reference_[k]['parts'][part_id] = reference_[
                    k]['parts'][part_id][:m.start()]
            if 'liplate' in reference_[k]['parts'][part_id]:
                reference_[k]['parts'][part_id] = reference_[
                    k]['parts'][part_id].replace('liplate', 'license plate')
    return reference_
