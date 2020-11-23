'''
General trainer developed for vision model development.
'''
import copy
import importlib
import inspect
import json
import logging
import math
import os
import random
import shutil
from datetime import datetime
from typing import Dict

import nni
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from .base_trainer import BaseTrainer
from .Utils.serialization import NumpyJSONEncoder

logger = logging.getLogger(__name__)

def _get_effective_batch_size(effective_batch):
    '''In a distributed setting, there may be variance between labels
    '''
    label_d = {}
    for minibatch in effective_batch:
        semseg_labels = minibatch['num_semseg_labels']
        points_labels = minibatch['num_point_labels']
        all_labels = semseg_labels
        all_labels.update(points_labels)
        all_labels['total'] = minibatch['num_labels']
        for k,v  in all_labels.items():
            if k not in label_d:
                label_d[k] = 0
            label_d[k] += v 
    return label_d['total'], label_d

def _move_to_device(batch, device):
    if torch.is_tensor(batch):
        result = batch.to(device)
    elif isinstance(batch, dict):
        result = {k: _move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        result = [_move_to_device(v, device) for v in batch]
    elif isinstance(batch, tuple):
        result = tuple(_move_to_device(v, device) for v in batch)
    else:
        result = batch
    return result


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, decay=0):
        self.val = val
        if decay:
            alpha = math.exp(-n/decay)  # exponential decay over 100 updates
            self.sum   = alpha * self.sum    + (1-alpha) * val * n
            self.count = alpha * self.count  + (1-alpha) * n
        else:
            self.sum   += val * n
            self.count += n
        self.avg = self.sum / self.count
            

class PisaTrainer(BaseTrainer):
    def __init__(self, config: Dict) -> 'PisaTrainer':
        super().__init__(config)
        self.task = self.set_up_task(config)
        logger.info(self._get_log_string(f"Initialized PisaTrainer."))
    
    ### Begin PrivFuncs: Below are private functions
    def _get_log_string(self, string):
        return f"Rank=[{self.rank}]: {string}"

    def _process_batch(self, batch):
        self.effective_batch.append(batch)
        if len(self.effective_batch) == self._config['grad_accumulation']:
            self.model.train()
            for criterion in self.criteria.values():
                criterion.train()
            acc_loss = {}
            effective_batch_size, acc_items_per_batch = _get_effective_batch_size(self.effective_batch)
            for minibatch_ix, minibatch in enumerate(self.effective_batch):
                minibatch['effective_batch_size'] = effective_batch_size
                minibatch['batch_sizes'] = acc_items_per_batch
                minibatch = _move_to_device(minibatch, self._config['device'])
                loss_d = self.task.train_step(self, minibatch)
                # collect and accumulate the losses and sample sizes from this mini-batch
                for key in loss_d:
                    if key not in acc_loss:
                        acc_loss[key] = 0
                    acc_loss[key] += loss_d[key]

            # Update model, step optimizers and lr_scheduler
            self.step(effective_batch_size)
            self.total_labels_seen += effective_batch_size
            # update losses and item counts of an effective batch to the AverageMeters

            for key in acc_loss:
                if key not in self.train_loss:
                    self.train_loss[key] = AverageMeter()
                self.train_loss[key].update(acc_loss[key], 1, self._config['log_frequency'])
            for key in acc_items_per_batch:
                if key not in self.train_items_per_batch:
                    self.train_items_per_batch[key] = AverageMeter()
                self.train_items_per_batch[key].update(acc_items_per_batch[key], 1)
            
            self.effective_batch = []

    def _initialize_scheduler(self, optimizer):
        scheduler_config = copy.deepcopy(self._config['lr_scheduler_config'])
        default_scheduler = 'WarmupISRScheduler'
        scheduler_type = scheduler_config.get('scheduler', default_scheduler)
        scheduler_config['optimizer'] = optimizer
        if scheduler_type == default_scheduler:
            scheduler_config['warmup_steps'] = scheduler_config.get('warmup_steps', 1000)
            scheduler_config['warmup_from_zero'] = scheduler_config.get('warmup_from_zero', False)

        try:  # first look for pytorch native lr scheduler
            lr_scheduler_class = getattr(lr_scheduler, scheduler_type)
            logger.info(f"Using pytorch native lr scheduler: {scheduler_type}")
        except:
            try:  # then look for custom lr scheduler inside pisa/Schedulers
                lr_scheduler_module = importlib.import_module(f"..Schedulers.{scheduler_type}", package=__name__)
                lr_scheduler_class = getattr(lr_scheduler_module, scheduler_type)
                logger.info(f"Using custom lr scheduler: {scheduler_type}")
            except Exception as e:
                logger.error(str(e))
                logger.error(f"ERROR: LR Scheduler {scheduler_type} is unknown")
                raise e
        scheduler_config.pop('scheduler', None)
        scheduler = lr_scheduler_class(**scheduler_config)
        self.lr_scheduler = scheduler

    def _initialize_optimizer(self):
        optimizer_config = self._config['optimizer_config']
        logger.info(f"Optimizer config: {optimizer_config}")
        optimizer_config['lr'] = optimizer_config.get('lr', self._config['lr_scheduler_config']['lr'])
        # instantiate optimizer for each module
        try:  # first try pytorch native optimizer
            optimizer_class = None
            if optimizer_config['optimizer'] == 'Adam':
                try:
                    from apex.optimizers.fused_adam import FusedAdam
                    optimizer_class = FusedAdam
                except:
                    pass
            if optimizer_class is None:
                optimizer_class = getattr(optim, optimizer_config['optimizer'])
                logger.info(f"Using pytorch native optimizer: {optimizer_config['optimizer']}")
        except:
            try:  # then try custom optimizer inside Models.Optimizers
                optimizer_module = importlib.import_module('..Optimizers.' + optimizer_config['optimizer'], package=__name__)
                optimizer_class = getattr(optimizer_module, optimizer_config['optimizer'])
                logger.info(f"Using custom optimizer: {optimizer_config['optimizer']}")
            except Exception as e: # then try deepspeed lamb optimizer
                logger.error(str(e))
                logger.error(f"ERROR: Optimizer {optimizer_config['optimizer']} is unknown")
                raise e
                    
        parameters = self.model.get_training_parameters()
        optimizer_config.pop('optimizer')
        logger.info(f"Initializing optimizer class: {optimizer_class} with config {optimizer_config}")
        self.optimizer = optimizer_class(parameters, **optimizer_config)
        self.optimizer.zero_grad()
        num_params = sum(p.numel() for group in self.optimizer.param_groups for p in group['params'])
        logger.info(f"Number of trainable parameters in modl: {num_params}")


    def _load_checkpoint(self, checkpoint_json_path=None):
        """
        Load complete training states, including model weights, optimizers, lr_schedulers,
        fp16 loss scaler, random state, batch generator, and updates count

        If 'checkpoint_json_path' is not given, this uses the default path.
        You can pass this path explicitly to allow cross-resuming into a new training folder.

        If 'must_exist' is False, then this will just return if the file does not exist,
        meaning to start over from start.
        """
        try:
            if checkpoint_json_path is None:
                checkpoint_json_path = os.path.join(self.artifact_folder, '..', 'resume_checkpoint.json')
            # find the checkpoint location and the tag from json file
            with open(checkpoint_json_path, encoding='utf-8') as f:
                checkpoint_location = json.load(f)  # @TODO: can we not pass the pathname directly?
            checkpoint_path = os.path.join(  # checkpoint_path in the JSON is relative to the JSON file
                os.path.dirname(checkpoint_json_path), checkpoint_location['checkpoint_path'], checkpoint_location['checkpoint_tag'])
            if not os.path.isdir(checkpoint_path):
                logger.info(f"Checkpoint path does not exist: {checkpoint_path}")
                logger.info(f"Continuing without loading checkpoint")
                return
                
            tag = checkpoint_location['checkpoint_tag']
            best_scores = checkpoint_location['best_scores']
            self.current_best_model_path = checkpoint_location['current_best_model_path']
            self.task.set_eval_best_scores(best_scores)
        except Exception as error:
            logger.error(f"Error loading checkpoint {error}")
            logger.info(f"Failed to load checkpoint JSON file: {checkpoint_json_path}")
            logger.info(f"Continuing without loading checkpoint")
            return

        # save a copy of the resumed checkpoint location in the save folder of current run
        if self.rank == 0:
            with open(os.path.join(self.save_folder, 'resumed_checkpoint.json'), 'w', encoding='utf-8') as f:
                json.dump(checkpoint_location, f)

        logger.warning(f'Loading checkpoint from {checkpoint_path}...')
       
        model_load_path = os.path.join(checkpoint_path, 'model_states.pt')
        state = torch.load(model_load_path, map_location=self._config['device'])
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.lr_scheduler.load_state_dict(state['lr_scheduler'])
               

        load_path = os.path.join(checkpoint_path, 'trainer_states.pt')
        trainer_state = torch.load(load_path, map_location='cpu')
        self.step_idx              = trainer_state['step_idx']
        self.total_labels_seen     = trainer_state['total_labels_seen']
        self.train_loss            = trainer_state['train_loss']
        self.train_items_per_batch = trainer_state['train_items_per_batch']
        
        # Using exponential smoothing for checkpoint averaging
        # if self._config.get('WEIGHT_SMOOTHING_BETA', 0) > 0:
        #     self.smoothed_parameters = {}
        #     for module_name in self.module_names:
        #         smoothed_param_path = os.path.join(checkpoint_path, module_name, 'smoothed_parameters.pt')
        #         if os.path.isfile(smoothed_param_path):
        #             self.smoothed_parameters[module_name] = torch.load(smoothed_param_path, map_location='cpu')
        #         else:
        #             logging.warning(f"Could not find smoothed_parameters for module {module_name}")

        random_state_path = os.path.join(checkpoint_path, 'random_state_{:04d}'.format(self.rank))
        random_state = torch.load(random_state_path, map_location='cpu')
        random.setstate(random_state['random'])
        np.random.set_state(random_state['numpy_random'])
        torch.set_rng_state(random_state['torch_random'])
        if self._config['cuda']:
            torch.cuda.set_rng_state(random_state['torch_cuda_random'], device=self._config['device'])

        logger.warning(self._get_log_string("No need to resume batch generator or batch generator is not checkpointable. Didn't load from checkpoint."))
        logger.warning(self._get_log_string(f'Finished loading checkpoint from {checkpoint_path}.'))

    def _save_checkpoint(self, tag):
        save_folder = os.path.join(self.artifact_folder, str(tag))
        logger.info(self._get_log_string(f"Saving checkpoint with tag={tag} to {save_folder}"))
        if self.rank == 0:
            os.makedirs(save_folder, exist_ok=True)
        num_tries=2
        for i in range(num_tries):
            if self.rank == 0:
                # TODO: Check apex amp to support half()
                # if self._config.get('half', False):
                #     TODO
                amp_state = None
                # Save model/optimizer/lr_scheduler
                model_save_file = os.path.join(save_folder, 'model_states.pt')
                state = {'model': self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict(),
                         'lr_scheduler': self.lr_scheduler.state_dict(),
                         'amp_state': amp_state}
                torch.save(state, model_save_file)
                # Save trainer state
                trainer_save_file = os.path.join(save_folder, 'trainer_states.pt')
                state = {'step_idx': self.step_idx,
                          'total_labels_seen': self.total_labels_seen,
                          'train_loss': self.train_loss,
                          'train_items_per_batch': self.train_items_per_batch}
                torch.save(state, trainer_save_file)
            random_save_file = os.path.join(save_folder, 'random_state_{:04d}.pt'.format(self.rank))
            random_state = {'random': random.getstate(),
                            'numpy': np.random.get_state(),
                            'torch': torch.get_rng_state(),
                            'torch_cuda': torch.cuda.get_rng_state(device=self._config['device']) if str(self._config['device']) != 'cpu' else None}
            torch.save(random_state, random_save_file)
            if self.rank == 0:
                # save the latest checkpoint location to json file
                checkpoint_location = {'checkpoint_tag': str(tag),
                                        'best_scores': self.task.get_eval_best_scores(),
                                        'current_best_model_path': self.current_best_model_path,
                                        'checkpoint_path': os.path.relpath(save_folder, start=os.path.join(self.artifact_folder, '..'))}
                resume_checkpoint_fp = os.path.join(self.artifact_folder, '..', 'resume_checkpoint.json')
                logger.info(f"Saving resumption file to {resume_checkpoint_fp}")
                with open(resume_checkpoint_fp, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_location, f)
        return save_folder

    def _load_model(self, load_path):
        model_load_path = os.path.join(load_path, 'model_states.pt')
        state = torch.load(model_load_path, map_location=self._config['device'])
        self.model.load_state_dict(state['model'])
        logger.info(f"Loaded state dict. Model:\n{self.model}")
        if self._config.get('USE_SMOOTHED_PARAMETERS', False):
            smoothed_param_path = os.path.join(load_path, 'smoothed_parameters.pt')
            if os.path.isfile(smoothed_param_path):
                smoothed_params = torch.load(smoothed_param_path)
                nn.utils.vector_to_parameters(smoothed_params, self.model.parameters())
            else:
                logger.warning(self._get_log_string(f"Could not find smoothed parameters for model at {load_path}. Continuing evaluation with the regular (not smoothed) checkpoint."))

    def _update_best_model(self, validation_scores, checkpoint_folder=None):
        """
        This function writes the current model in the best_model folder replacing their contents if any.
        The model contents are copied directly from checkpoint_folder, if checkpoint_folder is none
        it means the contents were not saved, on this case, the save_checkpoint method is called.
        """
        self.current_best_model_path = os.path.join(self.best_model_path, 'model')

        # Saving current update if it was not saved by the trainer
        if checkpoint_folder is None:
            checkpoint_folder = self._save_checkpoint(self.step_idx)

        logger.info(self._get_log_string(f'Updating best model with score {validation_scores}'))
        # copy checkpoint contents to best model folder
        if os.path.isdir(self.current_best_model_path):
            shutil.rmtree(self.current_best_model_path, ignore_errors=True)
        shutil.copytree(checkpoint_folder, self.current_best_model_path)

        with open(os.path.join(self.best_model_path, 'best_model_score.json'), 'w') as f:
            data = {'validation_scores': validation_scores, 'tag': self.step_idx}
            json.dump(data, f)

    def _write_scores_to_tensorboard(self, scores, prefix='eval_score'):
        for k, obj in scores.items():
            key = f"{prefix}_{k}"
            if isinstance(obj, dict):
                _key = key
                for subkey, subobj in obj.items():
                    key = f"{_key}_{subkey}"
                    self.write_to_tensorboard(key, subobj, self.total_labels_seen)
            else:
                    self.write_to_tensorboard(key, obj, self.total_labels_seen)
    ### End PrivFuncs

    ### Begin IntFuncs: Below are functions called internally or by the task
    def set_up_task(self, config):
        task_name = config['task']
        task_mod = importlib.import_module(f"..Tasks.{task_name}", package=__name__)
        task_cls = getattr(task_mod, task_name)
        logger.debug(self._get_log_string(f"Setting up task"))
        task = task_cls(config)
        logger.debug(self._get_log_string(f"Done setting up task"))
        return task
    
    def forward_pass(self, task_forward_f, minibatch):
        # TODO: Add support for AMP, etc.
        return task_forward_f(self, minibatch)

    def backward_pass(self, loss):
        # TODO: Add support for AMP, etc.
        loss.backward()
        return loss

    def step(self, batch_size=None):
        max_norm = self._config.get('max_gradient_norm', 0.0)
        if max_norm > 0.0:
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            if norm > max_norm:
                logger.info(self._get_log_string(f"Gradient was clipped. Norm: {norm} > {max_norm}"))
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()
        self.step_idx += 1
    ### End IntFuncs

    ### Begin UIFuncs: Below are functions called by the user interface (e.g., run.py)
    def train(self):
        self.current_best_model_path = None
        self.model = self.task.set_up_model()
        self.criteria = self.task.set_up_criteria()
        self.model.to(self._config['device'])        
        for criterion_name in self.criteria:
            self.criteria[criterion_name].to(self._config['device'])
        self.effective_batch = []
        self.set_up_tensorboard()        
        self.training_batch_generator = self.task.get_batch_generator(self, split_label='train')

        self.steps_per_epoch = len(self.training_batch_generator) 
        self.batches_per_epoch = self.steps_per_epoch * self._config['grad_accumulation']
        self.step_idx = 0
        self.total_labels_seen = 0
        self.train_loss = {}
        self.train_items_per_batch = {}
        self.batch_start_ix = 0

        self._initialize_optimizer()
        self._initialize_scheduler(self.optimizer)

        if self._config['resume'] != False:
            self._load_checkpoint(self._config['resume'] if isinstance(self._config['resume'], str) else None)

        logger.info(f"################### Beginning training ####################")
        logger.info(f"\tmodel type: {type(self.model)}")
        logger.info(f"\tcriterion type: {type(self.criteria)}")
        logger.info(f"\tEpoch length: {self.steps_per_epoch}")
        logger.info(f"###########################################################")

        trial_budget = int(self._config.get('TRIAL_BUDGET', 100))
        steps_trial_budget = (trial_budget / 100.0) * (self._config['num_epochs']* self.steps_per_epoch // self._config['grad_accumulation'])
        logger.info(f'Trial budget: {trial_budget} | steps_trial_budget: {steps_trial_budget}')
        start_epoch = self.batch_start_ix % self.steps_per_epoch
        end_epoch = self._config['num_epochs']
        for epoch in range(start_epoch, end_epoch):
            self.current_epoch_idx = epoch
            logger.info(self._get_log_string(f"Beginning training for epoch {epoch}"))
            epoch_start_time = datetime.now()
            for batch_idx, batch in enumerate(self.training_batch_generator):
                if int(self.step_idx) > steps_trial_budget:
                    logger.info(f"Steps trial budget of {steps_trial_budget} exceeded. Exiting trial.")
                    break
                self.current_batch_idx = self.batches_per_epoch * epoch + batch_idx
                # Fast-forward if necessary
                if batch_idx == start_epoch and self.current_batch_idx < self.batch_start_ix:
                    continue
                
                do_checkpoint = False
                do_evaluate = False
                # Only save/evaluate once per effective batch
                if (self.current_batch_idx + 1) % self._config['grad_accumulation'] == 0:
                    if self._config['save_frequency'] is not None and (self.step_idx + 1) % self._config['save_frequency'] == 0:
                        do_checkpoint = True
                    if self._config['eval_frequency'] is not None and (self.step_idx + 1) % self._config['eval_frequency'] == 0:
                        do_evaluate = True
                    elif self._config['eval_frequency'] is None:
                        do_evaluate = do_checkpoint
                
                _step_idx = self.step_idx
                self._process_batch(batch)
                
                if do_evaluate:
                    scores, is_best = self.task.evaluate_model(self, self._config.get('dev_split_name', 'val'), self.artifact_folder)
                    self._write_scores_to_tensorboard(scores)
                    nni.report_intermediate_result(scores)
                    logger.info(f"Done evaluating on step {self.step_idx}. Scores: {scores}")
                    if is_best:
                        logger.info(self._get_log_string(f"Current best score on rank-{self.rank} at step {self.step_idx}"))
                    if is_best:
                        self._update_best_model(scores)
                        do_checkpoint = True
                    if self._config.get('test_often', False):
                        logger.info(self._get_log_string(f"Evaluating on test_split on step {self.step_idx}."))
                        test_scores, _ = self.task.evaluate_model(self, self._config.get('test_split_name', 'test'), self.artifact_folder)
                        self._write_scores_to_tensorboard(test_scores, prefix='test_score')
                        logger.info(f"Done evaluating on test_split on step {self.step_idx}. Scores: {test_scores}")
                        # Do not use testing to determine best_model. That would bias the model selection process

                if do_checkpoint:
                    self._save_checkpoint(self.step_idx)
                if _step_idx != self.step_idx:
                    log_first = self._config.get("log_every_until", 10)
                    log_frequency = self._config.get('log_frequency', 100)
                    if (self.step_idx % log_frequency == 0) or (epoch == 0 and self.step_idx < log_first) or self._config.get('debug', False):
                        if hasattr(self.lr_scheduler, 'get_last_lr'):
                            last_lr = self.lr_scheduler.get_last_lr()[0]
                        else:
                            last_lr = self.lr_scheduler.get_lr()[0]

                        logger.info(self._get_log_string(f"epoch=[{epoch:6}] step_idx[{self.step_idx:.0f}] "
                                    f"learning_rate[{last_lr}] "
                                    f"training_loss[{', '.join([f'{key}: {obj.val:.5f}/{obj.avg:.5f}' for key, obj in self.train_loss.items()])}] "
                                    # f"items_per_second[{', '.join([f'{key}: {int(train_items_delta[key] / train_time_delta)}' for key in self.train_items_per_batch])}] "
                                    f"total_items[{', '.join([f'{key}: {obj.sum}' for key, obj in self.train_items_per_batch.items()])}] "
                                    f"epochs remaining[{str((datetime.now() - epoch_start_time) / (batch_idx  + 1) * (self.batches_per_epoch - batch_idx - 1)).split('.')[0]}]"))

                        for key, obj in self.train_loss.items():
                            self.write_to_tensorboard(f"train_loss_{key}", obj.val, self.total_labels_seen)
                        for key, obj in self.train_items_per_batch.items():
                            self.write_to_tensorboard(f"items_per_batch_{key}", obj.avg, self.total_labels_seen)
                            self.write_to_tensorboard(f"total_items_{key}", obj.sum, self.total_labels_seen)
                        self.write_to_tensorboard(f"learning_rate", last_lr, self.total_labels_seen)

                if self._config.get('debug', False) and self.step_idx > 10:
                    logger.info(self._get_log_string(f"Ending debug loop"))
                    break

                if int(self.step_idx) > steps_trial_budget:
                    logger.info(f"Steps trial budget of {steps_trial_budget} exceeded. Exiting trial.")
                    break

            logger.info(self._get_log_string(f"Epoch {epoch} complete after {datetime.now() - epoch_start_time}"))
            if not self._config.get('dont_evaluate', False):
                logger.info(self._get_log_string(f"Evaluating at the end of  epoch {epoch}."))
                scores, is_best = self.task.evaluate_model(self, self._config.get('dev_split_name', 'val'), self.artifact_folder)
                self._write_scores_to_tensorboard(scores)
                nni.report_intermediate_result(scores)
                logger.info(f"Done evaluating on step {self.step_idx}. Scores: {scores}")
                if is_best:
                    logger.info(self._get_log_string(f"Current best score on rank-{self.rank} at step {self.step_idx}"))
                if is_best:
                    self._update_best_model(scores)
                    do_checkpoint = True

            if self._config.get('test_often', False):
                logger.info(self._get_log_string(f"Evaluating on test_split at the end of  epoch {epoch}."))
                scores, is_best = self.task.evaluate_model(self, self._config.get('test_split_name', 'test'), self.artifact_folder)
                self._write_scores_to_tensorboard(scores, prefix='test_score')
                logger.info(f"Done evaluating on test_split on step {self.step_idx}. Scores: {scores}")
                # Do not use testing to determine best_model. That would bias the model selection process

            if do_checkpoint:
                self._save_checkpoint(self.step_idx)

            if self._config.get('debug', False) and self.step_idx > 10:
                    logger.info(self._get_log_string(f"Ending debug loop"))
                    break

        best_scores = self.task.get_eval_best_scores()
        nni.report_final_result(best_scores)
        # Save model at the end of training
        checkpoint_folder = self._save_checkpoint(f"final_model_{self.step_idx}")
        return self.best_model_path if self.best_model_path is not None else checkpoint_folder       
    
    def evaluate(self, splits='test'):
        """
        Perform evaluation
        Evaluate saved model(s) in self._config['model_checkpoint'] with the datasets in 'splits'.
        """
        splits = splits.split(',')
        visualize = self._config.get('visualize', False)
        label_prefix = self._config.get('label_prefix', None)
        logger.info(self._get_log_string('-----------------------------------------------'))
        logger.info(self._get_log_string("Evaluating model ... "))
        self.model = self.task.set_up_model()
        self.criteria = self.task.set_up_criteria()
        self.model.to(self._config['device'])        
        for criterion_name in self.criteria:
            self.criteria[criterion_name].to(self._config['device'])
        
        def score_file_name(ckpt):
            ckpt_run_dir = os.path.basename(os.path.dirname(ckpt))[4:]
            ckpt_upd_dir = os.path.basename(ckpt)
            return f"score_{ckpt_run_dir}_{ckpt_upd_dir}.json"

        # load trained model
        if 'model_checkpoint' in self._config:
            if os.path.isdir(self._config['model_checkpoint']):
                model_path = self._config['model_checkpoint']  # this is a directory, not a file
            else:
                model_path = os.path.join(self.artifact_folder, self._config['model_checkpoint'])  # this is a directory, not a file

            if 'min_checkpoint' in self._config or 'max_checkpoint' in self._config:
                if not os.path.isdir(model_path):
                    raise ValueError(f"Model directory not found: {model_path}")

                # enumerate all run_ folders in the model path
                run_folders = [os.path.join(model_path, x.name) for x in os.scandir(model_path) if x.is_dir() and x.name.startswith('run_')]

                # enumerate all ckpt folders
                ckpt_folders = [os.path.join(x, y.name) for x in run_folders for y in os.scandir(x) if y.is_dir() and y.name.isdecimal()]
                if ckpt_folders == []:
                    ckpt_folders = [os.path.join(model_path, x.name) for x in os.scandir(model_path) if x.is_dir() and x.name.isdecimal()]
                if ckpt_folders == []:
                    logger.info(self._get_log_string(f"No checkpoint in {model_path}."))
                    return

                # enumerate all ckpt folders in [min_checkpoint, max_checkpoint]
                min_checkpoint = int(self._config.get('min_checkpoint', 0))
                max_checkpoint = int(self._config.get('max_checkpoint', 10**6))
                ckpt_folders = [x for x in ckpt_folders if min_checkpoint <= int(os.path.basename(x)) <= max_checkpoint]
                if ckpt_folders == []:
                    logger.info(self._get_log_string(f"No checkpoints in range [{min_checkpoint}, {max_checkpoint}]."))
                    return

                # enumerate ckpt folders with saved module folders
                ckpt_folders = [x for x in ckpt_folders if os.path.isdir(os.path.join(x, self.module_names[0]))]
                if ckpt_folders == []:
                    logger.info(self._get_log_string(f"No saved module folders in checkpoint folders."))
                    return

                # enumerate all unfinished ckpt folders
                root_save_folder = os.path.dirname(self.artifact_folder)
                ckpt_folders = [x for x in ckpt_folders if not os.path.isfile(
                    os.path.join(root_save_folder, score_file_name(x)))]
                if ckpt_folders == []:
                    logger.info(self._get_log_string(f"All checkpoints are evaluated."))
                    return

                ckpt_folders.sort(key=lambda x: int(os.path.basename(x)))

                self.init_tb_writers()
                with torch.no_grad():
                    for ckpt in ckpt_folders:
                        ckpt_run_dir = os.path.basename(os.path.dirname(ckpt))[4:]
                        ckpt_upd_dir = os.path.basename(ckpt)
                        save_folder = os.path.join(self.artifact_folder, f"res_{ckpt_run_dir}_{ckpt_upd_dir}")

                        self._load_model(ckpt)

                        all_results = {}
                        for eval_dataset in splits:
                            self.task.reset_eval_best_scores()
                            label = f'{label_prefix}_{eval_dataset}' if label_prefix else eval_dataset
                            if visualize:
                                self.task.visualize(
                                    self, eval_dataset, self.artifact_folder, label)
                            else:
                                scores, is_best = self.task.evaluate_model(
                                    self, eval_dataset, self.artifact_folder, label=label)
                                logger.info(self._get_log_string("{0} scores: {1}".format(
                                    eval_dataset, scores)))
                                all_results[eval_dataset] = {
                                    'scores': scores
                                }
                        if not visualize and self.rank == 0:
                            all_results_path = os.path.join(root_save_folder, score_file_name(ckpt))
                            with open(all_results_path, 'w') as f:
                                logger.info(f"Storing all results to: {all_results_path}")
                                json.dump(all_results, f, indent=4, cls=NumpyJSONEncoder)
                self.close_tb_writers()
            else:
                if not os.path.isdir(model_path):
                    raise ValueError(f"Model directory not found: {model_path}")
                self._load_model(model_path)

                scores = None
                all_results = {}
                with torch.no_grad():
                    for eval_dataset in splits:
                        self.task.reset_eval_best_scores()
                        label = f'{label_prefix}_{eval_dataset}' if label_prefix else eval_dataset
                        if visualize:
                            self.task.visualize(
                                self, eval_dataset, self.artifact_folder, label)
                        else:
                            scores, is_best = self.task.evaluate_model(
                                self, eval_dataset, self.artifact_folder, label=label)
                            logger.info(self._get_log_string("{0} scores: {1}".format(
                                eval_dataset, scores)))
                            all_results[eval_dataset] = {
                                'scores': scores
                            }
                    if not visualize and self.rank == 0:
                        all_results_path = os.path.join(self.artifact_folder, "scores.json")
                        with open(all_results_path, 'w') as f:
                            logger.info(self._get_log_string(f"Storing all results to: {all_results_path}"))
                            json.dump(all_results, f, indent=4, cls=NumpyJSONEncoder)
        else:
            raise ValueError("model_checkpoint not found.")
        
        if not visualize:
            # For NNI, set the default score
            default_score = 0
            for score_d in all_results.values():
                default_score += score_d['scores']['default']
            default_score /= len(all_results)
            all_results['default'] = default_score
            return all_results
        return {}

    def infer(self):
        raise NotImplementedError
    ### End UIFuncs
