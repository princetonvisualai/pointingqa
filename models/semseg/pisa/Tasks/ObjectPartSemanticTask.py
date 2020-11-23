import copy
import importlib
import logging
import os
from collections import Iterable
from typing import Dict, Tuple

import torch
from torch import nn

from ..Evaluation.pascal_evaluation import (get_cmap,
                                            numpyify_logits_and_annotations,
                                            outputs_tonp_gt)
from ..pisa_trainer import PisaTrainer, _move_to_device

logger = logging.getLogger(__name__)
try:
    from mpi4py import MPI
except:
    logger.warning(f"mpi4py could not be installed. Multi-gpu training not supported")

class ObjectPartSemanticTask(object):
    def __init__(self, config: Dict) -> 'ObjectPartSemanticTask':
        logger.debug(f"Initializing task'")
        self._config = copy.deepcopy(config)
        self.use_semantic_segmentation = self._config.get('use_semantic_segmentation', True)
        self._task_config = self._config.pop('task_config', {})
        from ..Evaluation.ObjectPartEvaluator import ObjectPartEvaluator
        self.evaluator = ObjectPartEvaluator(self._config)
        from ..Dataloading.ObjectPartDataloader import ObjectPartDataFactory
        self.batch_gen_factory = ObjectPartDataFactory
        self._loss_weights = self._config.get('loss_weights', {
            'semseg': 1,
            'objpart': 1
        })
        self.eval_best_scores = {}
        # TODO: Add support for task composition
        # self._sub_tasks = []
        # for task in self._task_config.get('sub_tasks', []):
        #     self._sub_tasks[task] = trainer.set_up_task(self._config)

    # public funcs
    def get_eval_best_scores(self):
        if hasattr(self, 'eval_best_scores'):
            return copy.deepcopy(self.eval_best_scores)
        return {}

    def reset_eval_best_scores(self):
        self.set_eval_best_scores({})

    def set_eval_best_scores(self, scores):
        self.eval_best_scores = copy.deepcopy(scores)

    def _forward_func(self, trainer, batch):
        logger.debug(f"IMG={batch['img'].shape}, DEVICE={batch['img'].device}")
        output = trainer.model(batch)
        seg_log = output['segmentation_logits']
        points_log = output['points_logits']
        logger.debug(f"Done with forward pass")
        if self.use_semantic_segmentation:
            semseg_loss = trainer.criteria['SemSeg'](seg_log, batch['segmentation_target'])
        else:
            semseg_loss = None
        points_loss = trainer.criteria['ObjPart'](points_log, batch['point_target'])
        results = {'semseg_loss': semseg_loss, 'points_loss': points_loss}
        if output.get('bin_points_logits', None) is not None:
            binary_points_loss = trainer.criteria['ObjPart'](output['bin_points_logits'], batch['binary_point_target'])
            results['binary_points_loss'] = binary_points_loss
        return results
    
    @staticmethod
    def _forward_infer_func(trainer, batch):
        logger.debug(f"IMG={batch['img'].shape}, DEVICE={batch['img'].device}")
        output = trainer.model(batch)
        logger.debug(f"Done with forward pass")
        return output

    def train_step(self, trainer: PisaTrainer, minibatch) -> Tuple[Dict[str, float], Dict[str, int]]:
        all_labels = minibatch['batch_sizes']
        loss = trainer.forward_pass(self._forward_func, minibatch)
        semseg_loss = loss['semseg_loss']
        points_loss = loss['points_loss']
        binary_points_loss = loss.get('binary_points_loss', 0)
        if all_labels['points_total'] > 0:
            total_loss = (self._loss_weights['semseg'] * semseg_loss if semseg_loss is not None else 0.0) \
                        + self._loss_weights['objpart'] * points_loss \
                        + self._loss_weights.get('objpart_bin', self._loss_weights['objpart']) * binary_points_loss
        else:
            total_loss = self._loss_weights['semseg'] * semseg_loss
        
        trainer.backward_pass(total_loss)
        loss_output = {
            'total': total_loss.item() / all_labels['total'],
            'semseg': semseg_loss.item() / all_labels['semseg_total'] if semseg_loss is not None else -1.0
        }

        if all_labels['points_total'] > 0:
            loss_output['points'] = points_loss.item() / all_labels['points_total']
            if loss.get('binary_points_loss', None) is not None:
                loss_output['points_bin'] = binary_points_loss.item() / all_labels['points_total']

        logger.debug(f"Train step loss_output: {loss_output}")
        return loss_output

    def evaluate_model(self, trainer, split_label, save_folder, **kwargs):
        eval_batches = self.get_batch_generator(trainer, split_label)
        with torch.no_grad():
            scores, best_score = self.evaluator.evaluate_batches(trainer.model, eval_batches, dont_update_best=split_label=='test')
        if best_score:
            self.eval_best_scores = scores
        return scores, best_score


    def set_up_model(self) -> nn.Module:
        model_module = importlib.import_module(f"...Modeling.{self._config['model']}", package=__name__)
        model_class = getattr(model_module, self._config['model'])
        model = model_class(self._config)
        return model
    
    def set_up_criteria(self) -> Dict[str, nn.Module]:
        criteria = {}
        # We ignore criteria for now...
        # for criterion_name in self._config['criteria']:
        #     criterion_class = importlib.import_module(f"..Criteria.{criterion_name}", package=__name__)
        #     criterion = criterion_class(self._config)
        #     criteria[criterion_name] = criterion
        criteria['ObjPart'] = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=-1)
        criteria['SemSeg'] = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=255)
        return criteria

    def get_batch_generator(self, trainer, split_label):
        logger.debug(f"Getting {split_label} generator.")
        batch_generator = self.batch_gen_factory(self._config, 
                        label=split_label,
                        world_size=self._config['world_size'],
                        rank=self._config['rank'],
                        seed=trainer.seed)
        return batch_generator

    def visualize(self, trainer: PisaTrainer, dataset_label: str, save_dir: str, tag: str):
        ''' Output images showing the object/part inference results for a given input.
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
        24: "boat_part",
        25: "bottle_part",
        26: "bus_part",
        27: "car_part",
        28: "cat_part",
        29: "chair_part",
        30: "cow_part",
        31: "diningtable_part",
        32: "dog_part",
        33: "horse_part",
        34: "motorbike_part",
        35: "person_part",
        36: "pottedplant_part",
        37: "sheep_part",
        38: "sofa_part",
        39: "train_part",
        40: "tvmonitor_part"
        '''
        alpha = self._config.get('visualization_alpha', 0.7)
        from PIL import Image
        import numpy as np
        with torch.no_grad():
            batches = self.get_batch_generator(trainer, dataset_label)
            split_type = self._config.get('label_selection_criterion', 'all')
            save_dir = os.path.join(save_dir, tag, split_type)
            os.makedirs(save_dir, exist_ok=True)
            
            which = self._config.get('visualization_target', 'objpart')
            which = which.split(',')
            for cls_type in which:
                os.makedirs(os.path.join(save_dir, cls_type))
            # hardcoded in for the object-part infernce
            # no_parts = [0, 4, 9, 11, 18, 20, 24, 29, 31, 38, 40]
            # objpart_labels, semantic_labels = labels
            cmap = get_cmap()
            # valset_loader
            for i, minibatch in enumerate(batches):
                minibatch = _move_to_device(minibatch, self._config['device'])
                net_output = trainer.forward_pass(ObjectPartSemanticTask._forward_infer_func, minibatch)
                
                semantic_logits = net_output['segmentation_logits'].cpu()
                objpart_logits = net_output['points_logits'].cpu()
                semantic_anno = minibatch['segmentation_target'].cpu()
                objpart_anno = minibatch['point_target'].cpu()

                # First we do argmax on gpu and then transfer it to cpu
                predictions = {}
                if 'semantic' in which:
                    prediction, _ = numpyify_logits_and_annotations(
                        semantic_logits, semantic_anno, flatten=False)
                    predictions['semantic'] = prediction

                if 'separated' in which:
                    prediction, _ = numpyify_logits_and_annotations(
                        objpart_logits, objpart_anno, flatten=False)
                    predictions['separated'] = prediction
                
                if 'objpart' in which:
                    prediction, _ = outputs_tonp_gt(
                        objpart_logits, semantic_anno, trainer.model.op_map, flatten=False)
                    prediction[np.logical_and(
                        prediction > 0, prediction < 21)] = 1  # object
                    prediction[prediction > 20] = 2  # part
                    predictions['objpart'] = prediction

                if len(predictions) == 0:
                    raise ValueError(
                        '"which" value of {} not valid. Must be one of "semantic",'
                        '"separated", or'
                        '"objpart"'.format(which))

                for which_type, prediction in predictions.items():
                    image = minibatch['img'].cpu()
                    image_copy = np.array(image).squeeze(0).transpose(1, 2, 0)
                    image_copy = image_copy.astype(np.float32)
                    image_copy -= image_copy.min()
                    image_copy /= image_copy.max()

                    # image_copy*=255
                    prediction = prediction.squeeze(0)
                    cmask = np.zeros_like(image_copy, dtype=np.float32)
                    classes = np.unique(prediction)

                    # sz = prediction.size
                    for sem_class in classes:
                        if sem_class <= 0:
                            continue
                        ind = prediction == sem_class
                        cmask[ind, :] = cmap[sem_class]

                    cmask = cmask.astype(np.float32) / cmask.max()
                    ind = prediction > 0
                    image_copy[ind] = image_copy[ind] * \
                        (1.0 - alpha) + cmask[ind] * (float(alpha))
                    image_copy = image_copy - image_copy.min()
                    image_copy = image_copy / np.max(image_copy)
                    image_copy = image_copy * 255
                    image_copy = image_copy.astype(np.uint8)
                    image_copy = Image.fromarray(image_copy)
                    save_path = os.path.join(save_dir, which_type, f"validation_{i}.png")
                    image_copy.save(save_path)
                    image_copy.close()
                # hxwx(rgb)
                if (i+1) % 5 == 0:
                    logger.info(f"Processed image #{i}/{len(batches)}")
