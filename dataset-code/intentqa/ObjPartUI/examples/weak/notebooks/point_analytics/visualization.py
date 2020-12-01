import importlib
from PIL import ImageDraw
from PIL import ImageFont
import point_analytics
importlib.import_module('.utils', 'point_analytics')
importlib.import_module('.visualization', 'point_analytics')


def drawPoint(im, y, x):
    '''
    Draw a point at pixel (y,x).
    Returns a copy of image im with point drawn in.
    '''
    cp = im.copy()
    draw = ImageDraw.Draw(cp)
    draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill='red', outline='blue')
    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill='green', outline='blue')
    # draw.point((x, y), 'red')
    return cp


def get_ims_for_review(data, label=True):
    assert(isinstance(data, dict))
    assert('_' in data.keys()[0])
    if label:
        import json
    images = []
    for key, info in data.items():
        keydat = point_analytics.utils.separate_key(key)
        img = point_analytics.utils.get_im_from_s3(keydat['imid'])
        img = drawPoint(img, int(keydat['yCoord']), int(keydat['xCoord']))
        if label:
            txt = json.dumps(info)
            # font = ImageFont.truetype("arial.ttf", 16)
            font = ImageFont.truetype("/Library/Fonts/Times New Roman.ttf", 18)
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), "Resp: {}".format(txt), (0, 0, 0),
                      font=font)
        images.append((key, txt, img))

    return images
