Builds off of Pythia v0.3 (Credit: https://github.com/facebookresearch/mmf)

To run the Local-QA-model or Local-QA-grid-model (with computed features):

python3 -W ignore ./tools/run.py --tasks vqa --datasets vqamb --model pythia --config configs/vqa/vqamb/pythia.yml

To run the Local-QA-model on the IntentQA daraset:

python3 -W ignore ./tools/run.py --tasks vqa --datasets objpart --model pythia --config configs/vqa/objpart/pythia.yml

To run the Global-QA model on the LookTwice dataset:

python3 -W ignore ./tools/run.py --tasks vqa --datasets vqamb_level2 --model pythia_point --config configs/vqa/vqamb_level2/pythia_point.yml

The configs need to be set appropriately for these commands to succeed.
