import sys
import argparse
import json
import importlib
import numpy as np
from PIL import Image
import utils
import matplotlib.pyplot as plt
detail = importlib.import_module('../../../../../detail-api-3.0.0/'
                                 'PythonAPI/detail')


def blend_ims(im1, mask, typ=1):
    '''Blend 2 images together using different settings'''
    mask[mask == 255] = 0
    mask[mask > 0] = 255
    zee = np.zeros_like(mask)
    if typ == 1:
        mask2 = np.stack([zee, mask, zee], axis=2)
    else:
        mask2 = np.stack([zee, zee, mask], axis=2)
    img = Image.fromarray(mask2)
    return Image.blend(im1, img, alpha=0.2)


def get_points(cls, reference, details, num=10):
    '''Get all points from a click'''
    def onclick(event, obj, inst, part, imid, area, size, coords):
        '''Click handler'''
        global IX, IY
        print('you pressed', event.button, event.xdata, event.ydata)
        if int(event.button) == 3:
            print("Removing annotation {}".format(coords.pop()))
            return
        IX, IY = int(event.xdata), int(event.ydata)
        im_prefix = "https://s3.amazonaws.com/visualaipascalparts/"
        keydat = "{}_{}_{}".format(obj, inst, part)
        data = {imid: {"points": {"data": {keydat: {"points": [
            [IX, IY]], "area": area}}, "size": size},
                       "im": im_prefix + imid}}

        coords.append(data)
        print(coords[-1])

    def onLeave(event, fig, cid, coords):
        print('Leaving figure:\t{}'.format(event.key))
        key = event.key
        if key == 'backspace':
            print("Removing annotation {}".format(coords.pop()))
            return
        for click in cid:
            fig.canvas.mpl_disconnect(click)
        plt.close(fig)
        if key == 'q':
            sys.exit()

    anns = details.getAnns(cats=cls)[:num]
    i = 0
    all_coords = {}
    for j, ann in enumerate(anns):
        print("Displaying annotation {}".format(j))
        pid = 100
        inst_id = 0
        imid = str(ann['image_id'])
        imid = imid[:4] + "_" + imid[4:]
        img = utils.get_im_from_s3(imid)
        size = img.size
        size = size[0] * size[1]  # total image area...
        mask = details.decodeMask(ann['segmentation'])
        area = (mask > 0).sum()
        fig = plt.figure()
        axes = fig.add_subplot(111)
        cid = []
        coords = []
        onleave = utils.bake_function(onLeave, fig=fig, cid=cid, coords=coords)
        onclick = utils.bake_function(
            f=onclick,
            obj=cls,
            inst=inst_id,
            part=pid,
            imid=imid,
            area=area,
            coords=coords,
            size=size)
        img = blend_ims(img, mask, typ=1)
        axes.imshow(img)
        cid.append(fig.canvas.mpl_connect('button_press_event', onclick))
        # cid.append(fig.canvas.mpl_connect('figure_leave_event', onleave))
        cid.append(fig.canvas.mpl_connect('key_press_event', onleave))
        print("Part {} ({}) of area {}".format(
            pid, reference[str(cls)]['parts'][str(pid)], area))
        print("Showing annotation {}".format(i))
        plt.show()
        all_coords[i] = coords
        i += 1
        parts = ann['parts']
        for part in parts:
            pid = part['part_id']
            if int(pid) in [0, 255]:
                continue
            inst_id = 0
            mask = details.decodeMask(part['segmentation'])
            area = (mask > 0).sum()
            if area < 5:  # Not a real annotation
                continue
            fig = plt.figure()
            axes = fig.add_subplot(111)
            cid = []
            coords = []
            onleave = utils.bake_function(onLeave, fig=fig, cid=cid,
                                          coords=coords)
            onclick = utils.bake_function(
                f=onclick,
                obj=cls,
                inst=inst_id,
                part=pid,
                imid=imid,
                area=area,
                coords=coords,
                size=size)
            img = blend_ims(img, mask, typ=2)
            print("Part {} ({}) of area {}".format(
                pid, reference[str(cls)]['parts'][str(pid)], area))
            axes.imshow(img)
            cid.append(fig.canvas.mpl_connect('button_press_event', onclick))
            # cid.append(fig.canvas.mpl_connect('figure_leave_event', onleave))
            cid.append(fig.canvas.mpl_connect('key_press_event', onleave))
            print("Showing annotation {}".format(i))
            all_coords[i] = coords
            plt.show()
            i += 1

    return all_coords


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Annotate some class.')
    PARSER.add_argument(
        '-o',
        '--object',
        metavar='O',
        type=int,
        required=True,
        help='an PASCAL VOC object category (number). Valid numbers are 308,'
             '25, 59, 207, 23, 45, 416, 34, 98, 258, 347, 65, 113, 2,'
             'and 284.')
    PARSER.add_argument('-n', '--number', metavar='N', type=int, required=True,
                        help='Number of images you wish to annotate')

    ARGS = PARSER.parse_args()
    valid_classes = [
        308,
        25,
        59,
        207,
        23,
        45,
        416,
        34,
        98,
        258,
        347,
        65,
        113,
        2,
        284]
    if ARGS.object not in valid_classes:
        raise RuntimeError(
            "Invalid object class ID. {} not in {}".format(
                ARGS.object, valid_classes))
    with open('id_reference.json', 'r') as f:
        reference = json.load(f)

    annotation_file = '../../../../../JSON/trainval_parts.json'  # annotations
    image_dir = '../../../../../VOCdevkit/VOC2012/JPEGImages'  # jpeg images
    details = detail.Detail(annotation_file)
    coords = get_points(ARGS.object, reference, details, ARGS.number)
    print(coords)
    obj_name = reference[str(ARGS.object)]['name']
    with open('gt_points/' + obj_name + '.json', 'w+') as f:
        json.dump(coords, f)
