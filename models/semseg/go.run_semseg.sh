#!/bin/bash
experiment_name=nosemsup_run
python -m pisa.run 'train' --artifact-path artifacts/$experiment_name --data-path ./data/ --config configs/example_config.yml --overrides '{"save_frequency": 10, "eval_frequency": 10, "log_frequency": 1, "num_workers": 2}'
