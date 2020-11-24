import copy
import logging

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn

from ..Dataloading.PascalDataset import _get_pascal_class_names
from ..pisa_trainer import _move_to_device
from .pascal_evaluation import (compress_objpart_logits, get_accuracy, get_iou,
                                get_objpart_and_semantic_labels,
                                mask_logits_for_binary_pred,
                                numpyify_logits_and_annotations,)

logger = logging.getLogger(__name__)

try:
    from mpi4py import MPI
except:
    logger.warning(
        f"mpi4py could not be installed. Multi-gpu training not supported")




def _get_confusion_matrix(logits, anno, labels):
    pred_np, anno_np = numpyify_logits_and_annotations(logits, anno)
    if ((anno_np > 0).sum() == 0):
        return None
    cm = confusion_matrix(
        y_true=anno_np, y_pred=pred_np, labels=labels)
    return cm


def _evaluate_one(objpart_logits, semantic_logits, objpart_labels, semantic_labels, minibatch):
    semseg_target = minibatch['segmentation_target'].cpu()
    semseg_cm = _get_confusion_matrix(
        semantic_logits.cpu(), semseg_target, semantic_labels)
    objpart_target = minibatch['point_target'].cpu()
    objpart_logits_cpu = objpart_logits.cpu()
    objpart_cm = _get_confusion_matrix(
        objpart_logits_cpu, objpart_target, objpart_labels)

    masked_objpart_logits = mask_logits_for_binary_pred(
        objpart_logits_cpu, objpart_target)
    objpart_bin_cm = _get_confusion_matrix(
        masked_objpart_logits, objpart_target, objpart_labels)
    return objpart_cm, objpart_bin_cm, semseg_cm


class ObjectPartEvaluator(object):
    def __init__(self, config):
        self._config = copy.deepcopy(config)
        primary_eval_key = config.get('primary_eval_key', 'objpart_bin_accuracy')
        self.mask_type = config.get('mask_type', 'mode')
        self.valid_keys = ['semantic', 'objpart_bin', 'objpart',
                           'semantic_accuracy', 'objpart_accuracy', 'objpart_bin_accuracy']
        assert primary_eval_key in self.valid_keys
        self._primary_eval_key = primary_eval_key
        self.reset_best_score()
        self.best_res = {}

    def reset_best_score(self, **kwargs):
        self.best_score = {key: -float("Inf") for key in self.valid_keys}

    def _get_default_value(self, scores):
        return scores[self._primary_eval_key]

    def _check_is_best(self, scores):
        if scores[self._primary_eval_key] > self.best_score[self._primary_eval_key]:
            self.best_score = copy.deepcopy(scores)
            return True
        return False

    def evaluate_batches(self, model, batches, **kwargs):
        model.eval()
        if self.mask_type == 'soft':
            return self.evaluate_batches_kl_div(model, batches, **kwargs)
        return self.evaluate_batches_accuracy(model, batches, **kwargs)

    def evaluate_batches_kl_div(self, model, batches, **kwargs):
        kldiv_loss = nn.KLDivLoss(reduction='sum')
        scores = {}
        avg_loss = 0.0
        total_points = 0.
        with torch.no_grad():
            for minibatch_ix, minibatch in enumerate(batches):
                minibatch = _move_to_device(minibatch, self._config['device'])
                output = model(minibatch)
                # ignore binary logits and semantic logits if present
                # semantic_logits = output['segmentation_logits']
                objpart_logits = output['points_logits'].squeeze(0)
                objpart_logits = nn.functional.softmax(objpart_logits, dim=0)
                target = minibatch['point_target'].squeeze(0).permute(2, 0, 1)
                mask = target > -1
                objpart_logits = objpart_logits[mask]
                target = target[mask].float()
                loss = kldiv_loss(objpart_logits.log(), target)
                avg_loss += loss.item()
                total_points += (mask.sum().item() // 41)
        avg_loss /= total_points
        scores['default'] = avg_loss
        scores['kl_div'] = avg_loss
        return scores, False

    def evaluate_batches_accuracy(self, model, batches, **kwargs):
        scores = {}
        objpart_labels, semantic_labels = get_objpart_and_semantic_labels()
        total_objpart_cm, total_objpart_bin_cm, total_semseg_cm = None, None, None
        with torch.no_grad():
            for minibatch_ix, minibatch in enumerate(batches):
                minibatch = _move_to_device(minibatch, self._config['device'])
                output = model(minibatch)
                # ignore binary logits if present
                semantic_logits = output['segmentation_logits']
                objpart_logits = output['points_logits']
                objpart_cm, objpart_bin_cm, semseg_cm = _evaluate_one(
                    objpart_logits, semantic_logits, objpart_labels, semantic_labels, minibatch)
                if total_objpart_cm is None:
                    total_objpart_cm = objpart_cm
                    total_objpart_bin_cm = objpart_bin_cm
                else:
                    total_objpart_cm += objpart_cm
                    total_objpart_bin_cm += objpart_bin_cm
                if semseg_cm is not None:
                    if total_semseg_cm is None:
                        total_semseg_cm = semseg_cm
                    else:
                        total_semseg_cm += semseg_cm

        if self._config['world_size'] > 1:
            cms = [total_objpart_cm, total_objpart_bin_cm, total_semseg_cm]
            cms_flattened = []
            for cm in cms:
                cms_flattened += cm.flatten().tolist()
            cms_flattened = np.array(cms_flattened)
            cms_flattened = MPI.COMM_WORLD.allreduce(cms_flattened, op=MPI.SUM)
            _cms = []
            ix = 0
            for cm in cms:
                length = cm.size
                reduced_cm = np.array(
                    cms_flattened[ix:ix+length]).reshape(*cm.shape)
                _cms.append(reduced_cm)
                ix += length
            total_objpart_cm, total_objpart_bin_cm, total_semseg_cm = _cms
        # Semantic segmentation task
        semantic_accuracy = get_accuracy(total_semseg_cm)
        pascal_names = _get_pascal_class_names()
        semantic_IoU = get_iou(total_semseg_cm)
        semantic_mIoU = np.mean(semantic_IoU)
        semantic_all = {pascal_names[i]: v for i, v in enumerate(semantic_IoU)}

        no_parts = [0, 4, 9, 11, 18, 20, 24, 29, 31, 38, 40]
        objpart_accuracy = get_accuracy(total_objpart_cm)
        objpart_IoU = get_iou(total_objpart_cm)
        objpart_mIoU = np.mean(
            [opiou for i, opiou in enumerate(objpart_IoU) if i not in no_parts])
        objpart_all = {pascal_names[i]: v for i, v in enumerate(
            objpart_IoU) if i not in no_parts}
        objpart_bin_accuracy = get_accuracy(total_objpart_bin_cm)
        obj_bin_accuracy = get_accuracy(total_objpart_bin_cm[:21])
        part_bin_accuracy = np.diag(total_objpart_bin_cm[21:, 21:]).sum() \
                                    / total_objpart_bin_cm[21:].sum()
        objpart_bin_IoU = get_iou(total_objpart_bin_cm)
        objpart_bin_mIoU = np.mean(
            [opiou for i, opiou in enumerate(objpart_bin_IoU) if i not in no_parts])
        objpart_bin_all = {pascal_names[i]: v for i, v in enumerate(
            objpart_bin_IoU) if i not in no_parts}

        # Accuracy for each cls/part
        objpart_bin_accuracy_all = {pascal_names[i]: v[i] / v.sum() for i, v in enumerate(
            total_objpart_bin_cm) if i not in no_parts and v.sum() > 0.0}

        scores = {
            'objpart': objpart_mIoU,
            'objpart_bin': objpart_bin_mIoU,
            'semantic': semantic_mIoU,
            'objpart_all': objpart_all,
            'objpart_bin_all': objpart_bin_all,
            'semantic_all': semantic_all,
            'objpart_accuracy': objpart_accuracy,
            'objpart_bin_accuracy': objpart_bin_accuracy,
            'objpart_bin_accuracy_all': objpart_bin_accuracy_all,
            'obj_bin_accuracy': obj_bin_accuracy,
            'part_bin_accuracy': part_bin_accuracy,
            'semantic_accuracy': semantic_accuracy
        }

        scores['default'] = self._get_default_value(scores)
        if kwargs.get('dont_update_best', False):
            is_best = False
        else:
            is_best = self._check_is_best(scores)
        return scores, is_best
