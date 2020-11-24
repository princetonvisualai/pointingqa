import argparse
import copy
import json
import logging
from pathlib import Path

from .pisa_trainer import PisaTrainer
from .Utils.configuration_utils import validate_config
from .Utils.configuration_utils import get_config as _get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init(src=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('modes', type=str, help="Comma separated list including any of: train, test, infer")
    parser.add_argument('--artifact-path', type=Path, required=True, help="Path indicating where to store run artifacts (checkpoints, results, logs, etc.)")
    parser.add_argument('--data-path', type=Path, required=True, default=None, help='Path to the data file')
    parser.add_argument('--config', type=Path, required=False, default=None, help="Path to yaml or json configuration file.")
    parser.add_argument('--overrides', type=str, nargs='*', required=False, default=None, help="JSON string parameter to override the configuration parameters")
    parser.add_argument('--debug', default=False, action='store_true', help="Whether to run in a debugging setup")
    if src:
        return parser.parse_args(src)
    return parser.parse_args()

def get_config(args):
    config = _get_config(args.config, args.overrides)
    # Add cmdline arguments to save for reporting/checkpointing.
    for k, v in args.__dict__.items():
        config[k] = v
    validate_config(config)
    return config

def main(args, config=None):
    '''
    Main entry-point for training on Pisa.
    '''
    if config is None:
        config = get_config(args)
    config_copy = copy.deepcopy(config)
    for k, v in args.__dict__.items():
        config_copy[k] = str(v) # so it is JSON serializeable
    config_copy['device'] = str(config_copy['device'])
    logger.info(f"Starting `pisa.run` with config: {json.dumps(config_copy)}")
    
    modes = args.modes.split(',')
    
    # Use NNI to tune hyperparameters:
    if 'nni' in modes:
        import nni
        tuner_params = nni.get_next_parameter()
        artifact_suffix = '_nni_'
        for k, v in tuner_params.items():
            print(f"NNI {k} {v}")
            config[k] = v
            artifact_suffix += f"{k}-{v}_" 

        # Assist in nni
        if 'lr' in tuner_params:
            config['lr_scheduler_config']['lr'] = tuner_params['lr']
        if 'separate_heads' in tuner_params:
            config['model_config']['separate_heads'] = tuner_params['separate_heads']

        # Avoid resuming from different experiments
        config['artifact_path'] = Path(str(config['artifact_path']) + artifact_suffix)
        hp_opt = config.get('hp_optimizer', 'BOHB')
        assert(hp_opt != 'BOHB' or 'TRIAL_BUDGET' in config)

    trainer = PisaTrainer(config)

    if 'nni' in modes:       
        best_checkpoint_path = trainer.train()
        assert(best_checkpoint_path != None)
        trainer.update_config('model_checkpoint', best_checkpoint_path)
        # score = trainer.evaluate()
        # nni.report_final_result(score)
        return

    # Training
    if 'train' in modes:
        trainer.train()

    # No weight updating. Output predictions and report results
    if 'evaluate' in modes:
        splits = config.get('eval_splits', 'test')
        trainer.evaluate(splits)

    # Produce output but don't run an evaluation script
    if 'infer' in modes:
        trainer.infer()

if __name__ == '__main__':
    args = init()
    main(args)
