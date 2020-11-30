## Running Models

Builds off of Pythia v0.3 (Credit: https://github.com/facebookresearch/mmf). Please see this repo for questions about Pythia itself (e.g. config options).

To run the Local-QA-model or Local-QA-grid-model (with computed features):

```
python3 -W ignore ./tools/run.py --tasks vqa --datasets vqamb --model pythia --config configs/vqa/vqamb/pythia.yml
```

To run the Local-QA-model on the IntentQA daraset:

```
python3 -W ignore ./tools/run.py --tasks vqa --datasets objpart --model pythia --config configs/vqa/objpart/pythia.yml
```

To run the Global-QA model on the LookTwiceQA dataset:

```
python3 -W ignore ./tools/run.py --tasks vqa --datasets vqamb_level2 --model pythia_point --config configs/vqa/vqamb_level2/pythia_point.yml
```

The configs need to be set appropriately for these commands to succeed. Note the following models implemented apart from Pythia proper:

* **Pythia-point**: this is the implementation of the Global-QA-model. Requires both image and context features to be set.
* **Pythia-bbox**: this is an oracle method for LookTwice-QA which takes in the ground-truth bounding box around the point.
* **Pythia-noatt**: this model does not compute attention and assumes that only a single region proposal is provided (e.g. smallest or highest-scoring bounding box).
