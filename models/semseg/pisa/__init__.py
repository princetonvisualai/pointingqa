"""
Welcome to pisa, a flexible framework for training vision models.

The basic training command is:
    `python pisa.run train --config /path/to/config.yml --overrides "{'<arg1>': val1, ...}"`

Most parameters are provided in the configuration file with a few important exceptions:
    - `modes`: e.g., "train", "train,test" indicates the mode(s) the program should run in
    - `--artifact-path`: path indicating where to store run artifacts. This includes model checkpoints, predictions, etc.
    - `--data-path` (optional): root folder where the data is stored. Note that the trainer will also look for a data json/yaml file that outlines the relevant data files
The "run" script will check the `modes` argument to determine which program(s) to run (e.g., train ; train,test ; etc.) and then use the parameters stored in a configuration file to determine the rest.
"""