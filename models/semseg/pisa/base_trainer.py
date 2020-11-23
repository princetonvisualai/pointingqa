import copy
import logging
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from tensorboardX import SummaryWriter
from .Utils.configuration_utils import save_config

logger = logging.getLogger(__name__)

try:
    from mpi4py import MPI
except:
    logger.warning(f"mpi4py could not be installed. Multi-gpu training not supported")

def _get_environ_info():
    results = {}
    primary_address= '127.0.0.1'
    primary_port = '29500'
    if 'OMPI_COMM_WORLD_SIZE' not in os.environ:
        results['world_size'] = 1
        results['local_size'] = 1
        results['rank'] = 0
        results['local_rank'] = 0
        results['primary_address'] = primary_address
        results['primary_port'] = primary_port
    else:
        # Started with mpirun
        results['world_size'] = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        results['local_size'] = int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
        results['rank'] = int(os.environ['OMPI_COMM_WORLD_RANK'])
        results['local_rank'] = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        results['primary_address'] = primary_address
        results['primary_port'] = primary_port
    # TODO: Add support for multinode training
    return results

class BaseTrainer(object):
    def __init__(self, config: Dict) -> 'BaseTrainer':
        
        environ_info = _get_environ_info()
        for k, v in environ_info.items():
            config[k] = v
            setattr(self, k, v)
        self.backend_url = f'tcp://{self.primary_address}:{self.primary_port}'
        if config['cuda']:
            config['device'] = torch.device("cuda", config['local_rank'])
            torch.cuda.set_device(self.local_rank)
        else:
            assert self.world_size == 1, f"World size must be 1 to run using CPU"
            config['device'] = torch.device("cpu")
        self.seed = config.get('seed', 42)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        self._config = config
        self._get_artifact_directory()
        self._set_up_logging()
        self.save_config()
        if self.world_size > 1 or self._config['debug']:
            logger.info('Initializing process group')
            torch.distributed.init_process_group(backend='nccl',
                                                init_method=self.backend_url,
                                                world_size=self.world_size,
                                                rank=self.rank)
            logger.info('process group initialized')

    def _get_artifact_directory(self):
        run_num = 0
        artifact_directory = self._config['artifact_path'] # Path
        if self.rank == 0:
            if artifact_directory.exists() and not artifact_directory.is_dir():
                raise ValueError(f"``artifact-directory` either must indicate a path for a new dir to be created or a pre-existing directory path. {artifact_directory} exists but is not a directory.")
            elif not artifact_directory.exists():
                artifact_directory.mkdir(parents=True, exist_ok=False)
            existing_runs = Path(artifact_directory).glob('run_[0-9]*')
            for run in existing_runs:
                try:
                    _run_num = int(run.name.split('_')[-1])
                    if _run_num >= run_num:
                        run_num = _run_num + 1
                except:
                    continue
        artifact_folder = str(artifact_directory / f'run_{run_num}')
        if self.world_size > 1:
            run_num = MPI.COMM_WORLD.bcast(run_num, root=0)
            artifact_folder = str(artifact_directory / f'run_{run_num}')
        self.artifact_folder = artifact_folder
        self.best_model_path = os.path.join(artifact_folder, 'best_model')
        os.makedirs(self.best_model_path, exist_ok=True)
        self.run_num = run_num
        logger.info(f"Run artifacts being output to {self.artifact_folder}")


    def _set_up_logging(self):
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            logging_level = 'DEBUG' if self._config.get('debug', False) else self._config.get('logging_level', 'INFO')
            logging_level = getattr(logging, logging_level)
            # Allow debugging on all ranks. Suppress verbosity for normal training.
            if self.rank != 0 and not self._config.get('debug', False) :
                logging_level = max(logging_level, logging.WARNING)
            handlers = [logging.FileHandler(os.path.join(self.artifact_folder, f'log_{self.rank}.txt'), encoding='utf-8'), logging.StreamHandler()]

            logging.basicConfig(handlers=handlers,
                                format='%(asctime)-15s - %(levelname)s - %(name)s: %(message)s', 
                                datefmt='%Y.%m.%d %H:%M:%S', 
                                level=logging_level)
    def set_up_tensorboard(self):
        if self.rank == 0:
            tensorboard_folder = os.path.join(self.artifact_folder, 'tensorboard', 'log')
            self._tb_writer = SummaryWriter(log_dir=tensorboard_folder)
            logger.info(f"Tensorboard folder: {tensorboard_folder}")

    def update_config(self, key, value):
        self._config[key] = value

    def write_to_tensorboard(self, *args):
        if self.rank == 0:
            self._tb_writer.add_scalar(*args)
            self._tb_writer.flush()
    
    def close_tensorboard(self):
        self._tb_writer.close()

    def save_config(self):
        config_copy = {}
        for k, v in self._config.items():
            if isinstance(v, Path):
                config_copy[k] = str(v)
            else:
                config_copy[k] = v

        save_config(os.path.join(self.artifact_folder, 'config.yml'), config_copy)
        save_config(os.path.join(self.best_model_path, 'config.yml'), config_copy)

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError
    
    def infer(self):
        raise NotImplementedError