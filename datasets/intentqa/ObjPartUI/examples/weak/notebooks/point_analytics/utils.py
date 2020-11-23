'''Random utilities such as mapping to answers'''
import os
import functools
import json
from PIL import Image
from skimage import io
import numpy as np


class Average(object):
    def __init__(self, total=None, count=None):
        self.count = 0.0
        self.total = 0.0
        self.vals = []
        from numbers import Number
        self.Number = Number
        if isinstance(total, Number):
            self.total = total
            self.count = 1.0  # needs to be AT LEAST one
            self.vals = [total]
        elif isinstance(total, Average):
            self.total = total.total
            self.count = total.count
            self.vals = total.vals
        elif isinstance(total, list):
            self.vals = total
            self.total = sum(total)
            self.count = len(total)
        elif total is not None:
            raise ValueError(
                "Total's {} type not compatable.".format(
                    type(total)))

        if isinstance(count, Number):
            self.count = count
        elif count is not None:
            raise ValueError(
                "Count's {} type not compatable.".format(
                    type(count)))

    def toJSON(self):
        return json.dumps(self.vals)

    def std(self):
        return np.std(self.vals)

    def mean(self):
        return np.mean(self.vals)

    def append(self, val):
        self.count += 1
        self.total += val
        self.vals.append(val)

    def pop(self, ind=None):
        if ind is None:
            val = self.vals.pop()
        else:
            val = self.vals.pope(ind)
        self.total -= val
        self.count -= 1
        return val

    def as_number(self):
        return float(self)

    def __iadd__(self, other):
        self.append(other)
        return self

    def __isub__(self, other):
        self.pop(other)
        return self

    def __repr__(self):
        return str(self.__get__())

    def __format__(self, format_spec):
        '''Treat this as a float when printing'''
        return float(self.__get__()).__format__(format_spec)
        # if isinstance(format_spec, unicode):
        #     return unicode(str(self))
        # else:
        #     return str(self)

    def __get__(self, instance=None, owner=None):
        if self.count == 0:
            return 0.0
        return float(self.total / self.count)

    def __int__(self):
        return int(self.__get__())

    def __float__(self):
        return self.__get__()

    def __len__(self):
        return self.count

    def __add__(self, other):
        if isinstance(other, Average):
            total = self.total + other.total
            count = self.count + other.count
            return Average(total, count)
        elif isinstance(other, self.Number):
            return self + Average(other)
        else:
            raise ValueError(
                "{}'s {} type not supported".format(
                    other, type(other)))

    def __gt__(self, other):
        if isinstance(other, Average):
            return self.__get__() > other.__get__()
        else:
            return self.__get__() > other

    def __lt__(self, other):
        if isinstance(other, Average):
            return self.__get__() < other.__get__()
        else:
            return self.__get__() < other

    def __gte__(self, other):
        if isinstance(other, Average):
            return self.__get__() >= other.__get__()
        else:
            return self.__get__() >= other

    def __lte__(self, other):
        if isinstance(other, Average):
            return self.__get__() <= other.__get__()
        else:
            return self.__get__() <= other

    def __eq__(self, other):
        if isinstance(other, Average):
            return self.__get__() == other.__get__()
        else:
            return self.__get__() == other

    def __ne__(self, other):
        if isinstance(other, Average):
            return self.__get__() != other.__get__()
        else:
            return self.__get__() != other


def bake_function(f, **kwargs):
    return functools.partial(f, **kwargs)


def blend_ims(im1, im2, mask):
    pass


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1:
        p = p.flatten()

    return p


def map2each_answer(responses, f):
    for rsp in responses:
        output = rsp['output']
        for out in output:
            f(out)


def question2key(question):
    imid = list(question.keys())[0].strip()
    cls_inst_part = list(question[imid]['points']['data'].keys())[0]
    points = question[imid]['points']['data'][cls_inst_part]['points']
    x, y = str(points[0][0]), str(points[0][1])
    return imid + "_" + cls_inst_part + "_" + x + "_" + y


def separate_key(key):
    vals = key.rsplit("_", 5)
    return {
        'imid': vals[0],
        'obj_id': vals[1],
        'inst_id': vals[2],
        'part_id': vals[3],
        'xCoord': vals[4],
        'yCoord': vals[5]}


def forge_key(keydat):
    '''Takes kedyat modified output from ``separate_key`` and
    returns a new key.
    '''
    order = ['imid', 'obj_id', 'inst_id', 'part_id', 'xCoord', 'yCoord']
    key = [str(keydat[el]) for el in order]
    key = "_".join(key)
    return key


def get_im_from_s3(imid):
    if isinstance(imid, int):
        imid = str(imid)[:4] + "_" + str(imid)[4:]
    prefix = 'https://s3.amazonaws.com/visualaipascalparts/'
    suffix = '.jpg'
    uri = prefix + imid + suffix
    img = io.imread(uri)
    img = Image.fromarray(img)
    return img


def get_im_from_disk(imid):
    if isinstance(imid, int):
        imid = str(imid)[:4] + "_" + str(imid)[4:]
    prefix = os.path.dirname(os.path.realpath(__file__))
    navigate = os.path.join("..", "JPEGImages")
    print(prefix, navigate)
    prefix = os.path.normpath(os.path.join(prefix, navigate))
    print(prefix)
    suffix = '.jpg'
    uri = os.path.join(prefix, imid + suffix)
    img = io.imread(uri)
    img = Image.fromarray(img)
    return img


def intify_dict(dic_):
    ''' Converts all string number keys to integers.
    '''
    _dic = {}
    for k, v in dic_.items():
        v_ = v
        if isinstance(v, dict):
            v_ = intify_dict(v_)
        try:
            _dic[int(k)] = v_
        except ValueError:
            _dic[k] = v_
    return _dic


