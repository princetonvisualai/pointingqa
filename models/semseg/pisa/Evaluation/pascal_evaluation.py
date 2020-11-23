'''
author: --
with many functions adapted from @warmspringwinds
'''
import logging
import os
import shutil

import numpy as np
import torch
import torch.autograd
import torch.nn as nn
import torch.optim as optim
# Must be after to use locally modified torchvision libs
import torchvision
import torchvision.transforms
import tqdm
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable

# from .models import resnet_dilated
# from .models import partsnet
# from .models import objpart_net

logger = logging.getLogger(__name__)
# pylint: disable=too-many-arguments, invalid-name, len-as-condition,
# pylint: disable=too-many-locals, too-many-branches,too-many-statements
# pylint: disable=no-member

# PATH = os.path.dirname(os.path.realpath(__file__))
# PATHARR = PATH.split(os.sep)
# home_dir = os.path.join(
#     '/', *PATHARR[:PATHARR.index('obj_part_segmentation') + 1])
# VISION_DIR = os.path.join(home_dir, 'vision')
# DATASET_DIR = os.path.join(home_dir, 'datasets')
# sys.path.insert(0, home_dir)
# sys.path.insert(0, VISION_DIR)


def get_objpart_and_semantic_labels():
    number_of_classes = 41  # 20 * 2 + background (ignore some that lack parts)
    number_of_semantic_classes = 21  # 20 + background
    objpart_labels = range(number_of_classes)
    semantic_labels = range(number_of_semantic_classes)
    return list(objpart_labels), list(semantic_labels)


def poly_lr_scheduler(optimizer, init_lr, iteration, lr_decay_iter=1,
                      max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iteration is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power
        Credit @trypag
        https://discuss.pytorch.org/t/solved-learning-rate-decay/6825/5
    """
    if iteration % lr_decay_iter or iteration > max_iter:
        return optimizer

    lr = init_lr * (1 - iteration / max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def flatten_logits(logits, number_of_classes):
    """Flattens the logits batch except for the logits dimension"""
    logits_permuted = logits.permute(0, 2, 3, 1)
    logits_permuted_cont = logits_permuted.contiguous()
    logits_flatten = logits_permuted_cont.view(-1, number_of_classes)
    return logits_flatten


def get_valid_logits(logits, index, number_of_classes):
    ''' processes predictions based on the valid indices (selected
    from annotations)
    '''
    if len(index) == 0:
        return torch.Tensor([])
    logits_flatten = flatten_logits(
        logits, number_of_classes=number_of_classes)
    return torch.index_select(logits_flatten, 0, index)


def flatten_annotations(annotations):
    '''Literally just remove dimensions of tensor.
    '''
    return annotations.view(-1)


def get_valid_annotations_index(flat_annos, mask_out_value=255):
    ''' Returns a tensor of indices of all nonzero values
    in a flat tensor.
    '''
    nonz = torch.nonzero((flat_annos != mask_out_value))
    if nonz.numel() == 0:
        return torch.LongTensor([])
    return torch.squeeze(nonz, 1)


def get_valid_annos(anno, mask_out_value):
    ''' selects labels not masked out
        returns a flattened tensor of annotations and the indices which are
        valid
    '''
    anno_flatten = flatten_annotations(anno)
    index = get_valid_annotations_index(
        anno_flatten, mask_out_value=mask_out_value)
    if index.numel() == 0:
        return index.clone(), index
    anno_flatten_valid = torch.index_select(anno_flatten, 0, index)
    return anno_flatten_valid, index


def mask_logits_for_binary_pred(logits, anno, ignore_index=-1, num_objects=20):
    ''' process logits and annotations for input into
    confusion matrix function.

    args::
        ``logits``: network predictoins
        ``anno``: ground truth annotations

    returns::
        flattened predictions, flattened annotations
    '''

    def _get_corresponding_ix(val):
        if val <= num_objects:
            return val + num_objects
        return val - num_objects
    b, d, _, _ = logits.shape
    output = logits.clone().fill_(-100)
    unique_vals = anno.unique()
    for uv in unique_vals:
        if uv == ignore_index:
            continue
        mask = anno == uv
        maskshell = np.repeat(
            (mask.clone().fill_(False)).unsqueeze(1), d, 1).bool()
        # Fill in only the object and corresponding part
        ix = uv
        maskshell[:, ix, :, :] = mask
        ix2 = _get_corresponding_ix(ix)
        maskshell[:, ix2, :, :] = mask
        output[maskshell] = logits[maskshell]
    return output


def numpyify_logits_and_annotations(logits, anno, flatten=True):
    ''' process logits and annotations for input into
    confusion matrix function.

    args::
        ``logits``: network predictoins
        ``anno``: ground truth annotations

    returns::
        flattened predictions, flattened annotations
    '''
    # First we do argmax on gpu and then transfer it to cpu
    _logits = logits.data
    _, prediction = _logits.max(1)
    prediction = prediction.squeeze(1)
    prediction_np = prediction.cpu().numpy()
    anno_np = anno.numpy()
    if flatten:
        return prediction_np.flatten(), anno_np.flatten()
    return prediction_np, anno_np


def outputs_tonp_gt(logits, anno, op_map, flatten=True):
    ''' process logits and annotations for input into
    confusion matrix function.

    args::
        ``logits``: network predictoins
        ``anno``: ground truth annotations

    returns::
        flattened predictions, flattened annotations
    '''
    # def to_pair(ind, num_to_aggregate):
    #     '''Get the indices for corresponding obj-part pairs.'''
    #     if ind > num_to_aggregate:
    #         return [ind - num_to_aggregate, ind]
    #     return [ind, ind + num_to_aggregate]
    def to_pair(label, op_map):
        '''Use gt labels to select the correct pair of
           indices'''
        other_label = op_map[label]
        # new_label, pair = to_pair(label, 20)
        if label < other_label:
            return [label, other_label]
        return [other_label, label]

    _logits = logits.data.cpu()
    anno_np = anno.numpy()
    predictions = np.zeros_like(anno_np)
    # for index, anno_ind in np.ndenumerate(anno_np):
    for index in zip(*np.where(np.logical_and(anno_np > 0, anno_np != 255))):
        anno_ind = anno_np[index]
        batch_ind = index[0]
        i = index[1]
        j = index[2]
        channel_indices = to_pair(anno_ind, op_map)  # num_to_aggregate)
        aided_prediction = channel_indices[np.argmax(
            [_logits[batch_ind, ci, i, j] for ci in channel_indices])]

        predictions[batch_ind, i, j] = aided_prediction
    if flatten:
        return predictions.flatten(), anno_np.flatten()
    return predictions, anno_np


def compress_objpart_logits(logits, anno, op_map):
    ''' Reduce N x 41 tensor ``logits`` to an N x 2 tensor ``compressed``,
        where ``compressed``[0] => "object" and ``compressed``[1] => part
        (generic).
    args::
        ``logits``: network predictions => 2D tensor of shape (N, 41)
        ``anno``: ground truth annotations => 1D tensor of length N

    returns::
        compressed tensor of shape (N, 2)
    '''
    anno = anno.data.cpu()

    def to_pair(label, op_map):
        '''Use gt labels to isloate op loss
        '''
        other_label = op_map[label]
        # new_label, pair = to_pair(label, 20)
        if label < other_label:
            pair = [label, other_label]
            new_label = 0
        else:
            pair = [other_label, label]
            new_label = 1
        return new_label, pair

    indices = []
    new_anno = []
    for _, label in enumerate(anno):
        # new_label, pair = to_pair(label, 20)
        new_label, pair = to_pair(label, op_map)
        indices.append(pair)
        new_anno.append(new_label)
    len_ = len(indices)
    new_anno = Variable(torch.LongTensor(new_anno).cuda())
    indices = Variable(torch.LongTensor(indices).cuda())
    if len_ == 0:
        compressed_logits = Variable(torch.Tensor([]).cuda())
    else:
        compressed_logits = torch.gather(logits, 1, indices)
    return compressed_logits, new_anno


def get_accuracy(conf_mat):
    '''
    Used for computing the accuracy metric
    using a confusion matrix. .
    '''
    return np.diag(conf_mat).sum() / conf_mat.sum()


def get_iou(conf_mat, remove_empties=True):
    '''

    Used for computing the intersection over union metric
    using a confusion matrix. Pads unseen labels (union)
    with epsilon to avoid nan.
    Returns a vector of length |labels| with
    the IoU for each class in its appropriate
    place.

    '''
    intersection = np.diag(conf_mat)
    gt_set = conf_mat.sum(axis=1)
    predicted_set = conf_mat.sum(axis=0)
    union = gt_set + predicted_set - intersection
    
    if remove_empties:
        iou = []
        for inter, unio in zip(intersection, union):
            if unio == 0:
                continue
            iou.append(inter / float(unio))
        return np.array(iou)
    # Ensure no divide by 1 errors
    eps = 1  # 1e-5
    union[union == 0] = eps
    iou = intersection / union.astype(np.float32)
    return iou


def get_precision_recall(conf_mat):
    ''' Returns the class-wise precision and recall given a
        confusion matrix.
        Note that this defaults to 0 to avoids divide by zero errors.
    '''
    intersection = np.diag(conf_mat)
    gt_set = conf_mat.sum(axis=1)
    predicted_set = conf_mat.sum(axis=0)
    precision = intersection / \
        np.array([np.max([pred, 1.0]) for pred in predicted_set]).astype(
            np.float32)
    recall = intersection / \
        np.array([np.max([gt, 1.0]) for gt in gt_set]).astype(np.float32)
    return precision, recall


def validate_batch(
        objpart_dat,
        semantic_dat,
        overall_part_confusion_matrix,
        overall_semantic_confusion_matrix,
        labels,
        op_map,
        writer=None,
        index=0):
    ''' Computes the running IoU for the semantic and object-part tasks.
        args::
            :param (objpart_logits, objpart_anno): prediction, ground_truth
                    tensors for the object-part inference task
            :param (semantic_logits, semantic_anno): ditto for the semantic
            segmentation task
            :param overal_semantic_confusion_matrix: None or tensor of length
                                    |segmentation classes|. Total confusion
                                    matrix for semantic segmentation task
                                    for this epoch.
            :param overal_part_confusion_matrix: None or tensor of length
                                    |segmentation classes|. Total confusion
                                    matrix for object-part inference task
                                    for this epoch.

    '''
    (objpart_logits, objpart_anno) = objpart_dat
    (semantic_logits, semantic_anno) = semantic_dat
    objpart_labels, semantic_labels = labels
    semantic_prediction_np, semantic_anno_np = numpyify_logits_and_annotations(
        semantic_logits, semantic_anno)
    # objpart_prediction_np, objpart_anno_np = numpyify_logits_and_annotations(
    #     objpart_logits, objpart_anno)
    objpart_prediction_np, objpart_anno_np = outputs_tonp_gt(
        objpart_logits, objpart_anno, op_map)

    no_parts = [0, 4, 9, 11, 18, 20, 24, 29, 31, 38, 40]
    # Make sure to ignore all background class values
    objpart_anno_np[objpart_anno_np == 0] = -1
    # objpart_prediction_np[objpart_anno_np == 0] = -1

    # Mask-out value is ignored by default in the sklearn
    # read sources to see how that was handled
    current_semantic_confusion_matrix = confusion_matrix(
        y_true=semantic_anno_np,
        y_pred=semantic_prediction_np,
        labels=semantic_labels)

    if overall_semantic_confusion_matrix is None:
        overall_semantic_confusion_matrix = current_semantic_confusion_matrix
    else:
        overall_semantic_confusion_matrix += current_semantic_confusion_matrix
    try:
        current_objpart_confusion_matrix = confusion_matrix(
            y_true=objpart_anno_np, y_pred=objpart_prediction_np,
            labels=objpart_labels)

        if overall_part_confusion_matrix is None:
            overall_part_confusion_matrix = current_objpart_confusion_matrix
        else:
            overall_part_confusion_matrix += current_objpart_confusion_matrix

        objpart_prec, objpart_rec = get_precision_recall(
            current_objpart_confusion_matrix)
        objpart_mPrec = np.mean(
            [prec for i, prec in enumerate(objpart_prec) if i not in no_parts])
        objpart_mRec = np.mean(
            [rec for i, rec in enumerate(objpart_rec) if i not in no_parts])

    except ValueError:
        current_objpart_confusion_matrix = None
        objpart_prec, objpart_rec = None, None
        objpart_mPrec, objpart_mRec = None, None

    semantic_IoU = get_iou(
        current_semantic_confusion_matrix)
    semantic_mIoU = np.mean(semantic_IoU)

    if writer is not None:
        # writer.add_scalar('data/objpart_mIoU', objpart_mIoU, index)
        writer.add_scalar('data/semantic_mIoU', semantic_mIoU, index)
        writer.add_scalars('data/semantic_IoUs',
                           {'cls ' + str(i): v for i,
                            v in enumerate(semantic_IoU)},
                           index)
        if objpart_mPrec is not None:
            writer.add_scalar('data/objpart_mPrec', objpart_mPrec, index)
        if objpart_mRec is not None:
            writer.add_scalar('data/objpart_mRec', objpart_mRec, index)
        if objpart_prec is not None:
            writer.add_scalars('data/part_prec',
                               {'cls ' + str(i): v for i,
                                v in enumerate(objpart_prec)},
                               index)
        if objpart_rec is not None:
            writer.add_scalars('data/part_rec',
                               {'cls ' + str(i): v for i,
                                v in enumerate(objpart_rec)},
                               index)

    return ((objpart_mPrec, objpart_mRec),
            semantic_mIoU, overall_part_confusion_matrix,
            overall_semantic_confusion_matrix)
    # return objpart_mIoU, semantic_mIoU, overall_part_confusion_matrix,
    # overall_semantic_confusion_matrix


def save_checkpoint(state, is_best, folder='models',
                    filename='checkpoint.pth.tar'):
    ''' Saves a model
        args::
            :param ``staet``: dictionary containing training data.
            :param ``is_best``: boolean determining if this represents
                            the best-trained model of this session
            :param ``folder``: relative path to folder in which to save
            checkpoint
            :param ``filename``: name of the checkpoint file

        additionally copies to "[architecture]" + "_model_best.pth.tar"
        if is_best.
    '''
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(
                folder, filename), os.path.join(
                    folder, state['arch'] + '_model_best.pth.tar'))


def load_checkpoint(load_path, fcn, optimizer):
    ''' Loads network parameters (and optimizer params) from a checkpoint file.
        args::
            :param ``load_path``: string path to checkpoint file.
            :param ``fcn``: torch.nn network
            :param ``optimizer``: duh
        returns the starting epoch and best scores
    '''
    if os.path.isfile(load_path):
        logger.info("=> loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path)

        try:
            start_epoch = checkpoint['epoch']
        except KeyError:
            start_epoch = 0
        try:
            best_semantic_val_score = checkpoint['best_semantic_mIoU']
            best_objpart_val_score = checkpoint['best_objpart_mIoU']
        except KeyError:
            best_semantic_val_score = 0.0
            best_objpart_val_score = 0.0
        try:
            best_objpart_accuracy = checkpoint['best_objpart_accuracy']
        except KeyError:
            best_objpart_accuracy = 0.0

        state_dict = {}
        model_sd = fcn.state_dict()

        if 'state_dict' in checkpoint:
            it = checkpoint['state_dict'].items()
        else:
            it = checkpoint.items()

        for i, (k, v) in enumerate(it):
            k_ = k.split(".")
            if k_[0] == 'resnet34_8s':
                k_[0] = 'net'
            elif 'layer' in k_[0]:
                k_.insert(0, 'net')
            k_ = ".".join(k_)
            if k_ not in model_sd:
                logger.warning(
                    "Layer {} from checkpoint not found in model".format(k_))
                continue
            elif model_sd[k_].size() != v.size():
                logger.warning(
                    "{}: {} not equal to {}".format(
                        i,
                        model_sd[k_].size(),
                        v.size()))
                continue
            else:
                state_dict[k_] = v
        # optim_state_dict[k_] = checkpoint['optimizer'][k]
        # fcn.load_state_dict(checkpoint['state_dict'])

        fcn.load_state_dict(state_dict, strict=False)
        if optimizer is not None:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except ValueError as e:
                logger.error("{}".format(e))
            except KeyError:
                ogger.error("optimizer not found in checkpoint")
        _epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
        ogger.info("=> loaded checkpoint '{}' (epoch {})"
                   .format(load_path, _epoch))
    else:
        raise RuntimeError("{} does not exist.".format(load_path))

    return (start_epoch, best_semantic_val_score,
            best_objpart_val_score, best_objpart_accuracy)


def get_cmap():
    ''' Return a colormap stored on disk
    '''
    import json
    fname = os.path.join(
        os.path.dirname(
            os.path.realpath(__file__)),
        'colortable.json')
    with open(fname, 'r') as f:
        cmap = json.load(f)
    return cmap


def validate_and_output_images(net, loader, op_map,
                               which='semantic', alpha=0.6):
    ''' Computes mIoU for``net`` over the a set.
        args:: :param ``net``: network (in this case resnet34_8s_dilated
            :param ``loader``: dataloader (in this case, validation set loader)

        returns the mIoU (ignoring classes where no parts were annotated) for
        the semantic segmentation task and object-part inference task.
      0: "background",
      1: "aeroplane",
      2: "bicycle",
      3: "bird",
      4: "boat",
      5: "bottle",
      6: "bus",
      7: "car",
      8: "cat",
      9: "chair",
      10: "cow",
      11: "diningtable",
      12: "dog",
      13: "horse",
      14: "motorbike",
      15: "person",
      16: "pottedplant",
      17: "sheep",
      18: "sofa",
      19: "train",
      20: "tvmonitor"
      21: "aeroplane_part",
      22: "bicycle_part",
      23: "bird_part",
      #24: "boat_part",
      24 - 25: "bottle_part",
      25 - 26: "bus_part",
      26 - 27: "car_part",
      27 - 28: "cat_part",
      29: "chair_part",
      28 - 30: "cow_part",
      31: "diningtable_part",
      29 - 32: "dog_part",
      30 - 33: "horse_part",
      31 - 34: "motorbike_part",
      32 - 35: "person_part",
      33 - 36: "pottedplant_part",
      34 - 37: "sheep_part",
      38: "sofa_part",
      35 - 39: "train_part",
      40: "tvmonitor_part"

    '''
    from PIL import Image
    net.eval()
    # hardcoded in for the object-part infernce
    # no_parts = [0, 4, 9, 11, 18, 20, 24, 29, 31, 38, 40]
    # objpart_labels, semantic_labels = labels
    cmap = get_cmap()

    # valset_loader
    i = 0
    for image, semantic_anno, objpart_anno in tqdm.tqdm(loader):
        img = Variable(image.cuda())
        objpart_logits, semantic_logits = net(img)

        # First we do argmax on gpu and then transfer it to cpu
        if which == 'semantic':
            prediction, _ = numpyify_logits_and_annotations(
                semantic_logits, semantic_anno, flatten=False)
        elif which == 'separated':
            prediction, _ = numpyify_logits_and_annotations(
                objpart_logits, objpart_anno, flatten=False)
        elif which == 'objpart':
            # prediction, anno = outputs_tonp_gt(
            #     objpart_logits, objpart_anno,semantic_anno, flatten=False)
            prediction, _ = outputs_tonp_gt(
                objpart_logits, semantic_anno, op_map, flatten=False)
            prediction[np.logical_and(
                prediction > 0, prediction < 21)] = 1  # object
            prediction[prediction > 20] = 2  # part
        else:
            raise ValueError(
                '"which" value of {} not valid. Must be one of "semantic",'
                '"separated", or'
                '"objpart"'.format(which))

        image_copy = np.array(image).squeeze(0).transpose(1, 2, 0)
        image_copy = image_copy.astype(np.float32)
        image_copy -= image_copy.min()
        image_copy /= image_copy.max()
        # image_copy*=255
        prediction = prediction.squeeze(0)
        cmask = np.zeros_like(image_copy, dtype=np.float32)
        classes = np.unique(prediction)
        # sz = prediction.size
        for cls in classes:
            if cls <= 0:
                continue
            ind = prediction == cls
            cmask[ind, :] = cmap[cls]

        cmask = cmask.astype(np.float32) / cmask.max()
        ind = prediction > 0
        image_copy[ind] = image_copy[ind] * \
            (1.0 - alpha) + cmask[ind] * (float(alpha))
        image_copy = image_copy - image_copy.min()
        image_copy = image_copy / np.max(image_copy)
        image_copy = image_copy * 255
        image_copy = image_copy.astype(np.uint8)
        image_copy = Image.fromarray(image_copy)
        image_copy.save("predictions/validation_{}_{}.png".format(which, i))
        i += 1
        image_copy.close()
        # hxwx(rgb)
