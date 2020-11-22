# Copyright (c) Facebook, Inc. and its affiliates.
from pythia.common.registry import registry
from pythia.tasks import BaseTask


@registry.register_task("vqa")
class VQATask(BaseTask):
    def __init__(self):
        super(VQATask, self).__init__("vqa")

    def _get_available_datasets(self):
        return ["vqa2", "vizwiz", "textvqa", "vqa2_ocr", "clevr", "visual_genome", "vqamb", "vqamb_level2", "verbal_spatial", "objpart", "localqa"]

    def _preprocess_item(self, item):
        return item
