# Pythia model, end-to-end applying detectron and resnet

from pythia.common.registry import registry
from pythia.models.base_model import BaseModel
from pythia.models.pythia import Pythia

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model

import torch
import torchvision.models as models
import torch.nn as nn

@registry.register_model("pythia_full")
class PythiaFull(BaseModel):

    def __init__(self, config):
        super().__init__(config)

    def build(self):
        self.pythia_model = self._build_pythia_model()

        # self.detectron_model = self._build_detectron_model()

        self.resnet152_model = self._build_resnet_model()

    def _build_pythia_model(self):
        model_config = self.config
        model = Pythia(model_config)
        
        model.build()
        model.init_losses_and_metrics()

        # load Pythia file in (hardcode for now)
        weights_path = '/n/fs/visualai-scr/arjuns/vqa/exp/pythia/pythia/save/vqa_vqamb_pythia/pythia_resnet_final.pth'

        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)

        # model.eval()
        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout):
                module.p = 0
                
            elif isinstance(module, nn.LSTM):
                module.dropout = 0

            elif isinstance(module, nn.GRU):
                module.dropout = 0
                        
        return model
        
    def _build_resnet_model(self):
        resnet152 = models.resnet152(pretrained=True)
        resnet152.eval()
        modules = list(resnet152.children())[:-2]

        resnet152_model = torch.nn.Sequential(*modules)
        resnet152_model.eval() # keep resnet parameters fixed
        return resnet152_model

    # Images shape: (batch_size, 3, W, H)
    def get_resnet_features(self, imgs):
        features = self.resnet152_model.forward(imgs).permute(0, 2, 3, 1)
        features = features.view(imgs.shape[0], 196, 2048)
        return features

    # TODO
    # ------
    def _build_detectron_model(self):
        pass

    def get_detectron_feat(self, imgs):
        pass
    # ------

    # Fields of sample: 'question', 'answer', 'points'.
    # Detectron: either 'detectron_img' or 'image_feature_0'
    # Resnet: either 'resnet_img' or image_feature_1'
    def forward(self, sample_list):

        fields = sample_list.fields()

        # TODO: for now assume the field exists
        # if 'image_feature_0' not in fields:

        # If so, calculate image features from resnet
        if 'image_feature_1' not in fields:
            resnet_img = sample_list.resnet_img
            resnet_feat = self.get_resnet_features(resnet_img)
            sample_list.add_field('image_feature_1', resnet_feat)
        
        scores = self.pythia_model.forward(sample_list)
        return scores
