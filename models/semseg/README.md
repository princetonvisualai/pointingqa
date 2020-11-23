Description of model

We get similar performance using ResNet101 and ResNet32. To replicate the ResNet101 results, run the file go.run_semseg.sh. This uses a ResNet101 model pretrained on ImageNet. To see training progress, we recommend using tensorboardX, though you can also observe the resulting log file in the specified artifact directory.

Running this command should automatically download the Pascal VOC dataset from the appropriate link. If you are having issues downloading, please copy the benchmark_RELEASE and Pascal_VOC files to the data folder of your experiment run (e.g. ./data).

Pisa also supports using [nni](https://github.com/Microsoft/nni) to automate hyperaparameter search. To use this, create a search_space.json file as specified in the nni README as well as an e.g. bohb config to specify the behavior. NNI can help automate tuning in parallel.