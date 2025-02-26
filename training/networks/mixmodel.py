'''
this model has 2 branches:
- model_1: pretrained-efnb4 on sbi, FROZEN
  + weight: logs/training/fsbi/fsbi_2025-02-05-20-57-18_89.87_res380+effnetb4+adam+linear/test/Celeb-DF-v2/ckpt_best.pth
- model_2: efnb4
'''

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .efficientnetb4 import EfficientNetB4
from metrics.registry import BACKBONE

logger = logging.getLogger(__name__)



@BACKBONE.register_module(module_name="mixmodel")
class MixModel(nn.Module):
    def __init__(self, config):
        super(MixModel, self).__init__()
        self.model_1 = EfficientNetB4(config["model_1"])
        self.model_2 = EfficientNetB4(config["model_2"])
        self.num_classes = config["num_classes"]
        self.last_layer = nn.Linear(1792 + 1792, self.num_classes)

        freeze_finetune_path_1 = config.get("freeze_finetune_path_1", "")
        if freeze_finetune_path_1 != "":
            saved = torch.load(freeze_finetune_path_1)
            suffix = freeze_finetune_path_1.split('.')[-1]
            if suffix == 'p':
                saved = saved.state_dict()
            saved = {k.replace("backbone.", ""): v for k, v in saved.items()}
            self.model_1.load_state_dict(saved)
            logger.info(f"Loaded pretrained model 1 from {freeze_finetune_path_1}")

        self.model_1.last_layer = nn.Identity()
        self.model_2.last_layer = nn.Identity()

        for param in self.model_1.parameters():
            param.requires_grad = False

    def features(self, input):
        x1 = self.model_1.features(input)
        x2 = self.model_2.features(input)
        return torch.cat([x1, x2], dim=1)

    def classifier(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        out = self.last_layer(x)
        return out

    def forward(self, input):
        x = self.features(input)
        out = self.classifier(x)
        return out


if __name__ == "__main__":
    config = {
        "model_1": {
            "mode": "original",
            "num_classes": 2,
            "inc": 3,
            "dropout": False
        },
        "model_2": {
            "mode": "original",
            "num_classes": 2,
            "inc": 3,
            "dropout": False
        },
        "freeze_finetune_path_1": "logs/training/fsbi/fsbi_2025-02-05-20-57-18_89.87_res380+effnetb4+adam+linear/test/Celeb-DF-v2/ckpt_best.pth",
        "pretrained_path": "./training/pretrained/efficientnet-b4-6ed6700e.pth",
        "num_classes": 2
    }

    model = MixModel(config)
    print(model)

    input = torch.randn(1, 3, 380, 380)
    out = model(input)
    print(out.shape)
    print(out)
