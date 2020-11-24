import json
from pathlib import Path

import ruamel.yaml as yaml
import torch


def get_config(config_path, cmdline_overrides):
    config = {}
    if config_path is not None:
        config = parse_config_file(config_path)
    if cmdline_overrides is not None:
        cmdline_overrides = ' '.join(cmdline_overrides)
        cmdline_params = json.loads(cmdline_overrides)
        for k, v in cmdline_params.items():
            config[k] = v
    return config

def parse_config_file(path: Path) -> dict:
    ''' Parse json or yaml configuration file.
        args:
            path (Path): path to the configuration file
    '''
    if not path.exists():
        assert FileExistsError(f"Path {str(path)} not found.")
    if '.yaml' in path.suffixes or '.yml' in path.suffixes:
        data = yaml.safe_load(path.open(encoding='utf-8'))
    elif '.json' in path.suffixes:
        data = json.load(path.open(encoding='utf-8'))
    else:
        assert ValueError(f"Path {str(path)} must be of type yaml (.yml or .yaml) or json (.json).")
    return data


def validate_config(config):
    config['resume'] = config.get('resume', True) # Could also be a str path indicating where to resume from
    config['cuda'] = config.get('cuda', True) and torch.cuda.is_available()
    config['device'] = torch.device('cuda' if config['cuda'] else 'cpu')
    config['half'] =  config.get('half', False)
    config['grad_accumulation'] = int(config.get('grad_accumulation', 1))
    config['save_frequency'] = config.get('save_frequency', None) # None means save every epoch (if defined in the dataloader)
    config['eval_frequency'] = config.get('eval_frequency', None) # None means eval only after training is completed
    config['criteria'] = [config.pop('criterion')] if 'criterion' in config else config['criteria']
    config['num_epochs'] = config.get('num_epochs', 10000)
    
    assert isinstance(config['criteria'], list) and isinstance(config['criteria'][0], str), f"Unsported Criteri[on|a]: pisa supports a string criterion parameter or a list of string criteria parameters. Got: {config['criteria']}"


def save_config(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f)