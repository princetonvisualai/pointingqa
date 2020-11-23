# Copyright (c) Facebook, Inc. and its affiliates.
import torch

from pythia.common.registry import registry
from pythia.models.pythia import Pythia
from pythia.modules.layers import ClassifierLayer

@registry.register_model("pythia_global")
class Pythia_global(Pythia):
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        self._init_text_embeddings("text")
        self._init_feature_encoders("context")
        self._init_feature_embeddings("context")
        super().build()

    def get_optimizer_parameters(self, config):
        params = super().get_optimizer_parameters(config)
        params += [
            {"params": self.context_feature_embeddings_list.parameters()},
            {
                "params": self.context_feature_encoders.parameters(),
                "lr": (config["optimizer_attributes"]["params"]["lr"] * 0.1),
            },
        ]

        return params

    def forward(self, sample_list):

        sample_list.text = self.word_embedding(sample_list.text)
        text_embedding_total = self.process_text_embedding(sample_list)

        context_embedding_total, att_pt = self.process_feature_embedding(
            "context", sample_list, text_embedding_total
        )

        image_embedding_total, att_img = self.process_feature_embedding(
            "image", sample_list, text_embedding_total, context=context_embedding_total
        )

        joint_embedding = self.combine_embeddings(
            ["image", "text"],
            [image_embedding_total, text_embedding_total, context_embedding_total],
        )

        scores = self.calculate_logits(joint_embedding)

        return {"scores": scores, "att_img": att_img[0].squeeze(), "att_pt": att_pt[0].squeeze()}
