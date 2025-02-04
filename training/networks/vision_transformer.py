# Just a note: https://github.com/lucidrains/vit-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import Union
from metrics.registry import BACKBONE
import os

@BACKBONE.register_module(module_name="vision_transformer")
class VisionTransformer(nn.Module):
    def __init__(self, vit_config):
        super(VisionTransformer, self).__init__()
        """ Constructor
        Args:
            vit_config: configuration file with the dict format
        """
        self.num_classes = vit_config["num_classes"]
        #self.inc = vit_config["inc"]
        #self.dropout = vit_config["dropout"]
        self.model_type = vit_config["model_type"]
        assert self.model_type in ['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14']

        # Load the ViT model without pre-trained weights
        if vit_config["pretrained"]:
            self.vit = getattr(torchvision.models, self.model_type)(image_size=vit_config["vit_resolution"])
        else:
            self.vit = getattr(torchvision.models, self.model_type)(image_size=vit_config["vit_resolution"], weights="DEFAULT")

        # Remove the last layer (the classifier)
        self.vit.heads = nn.Identity()

        ## Initialize the last_layer layer
        self.last_layer = nn.Linear(768, self.num_classes)

        for param in self.vit.parameters():
            param.requires_grad = True

        for param in self.last_layer.parameters():
            param.requires_grad = True

    def features(self, x):
        # Extract features
        x = self.vit(x)
        return x

    def classifier(self, x):
        x = x.view(x.size(0), -1)
        # Apply last_layer layer
        self.last_emb = x
        y = self.last_layer(x)
        return y

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

'''
comment before running:
- from metrics.registry import BACKBONE
- @BACKBONE.register_module(module_name="vision_transformer")
'''
if __name__ == '__main__':
    vit_config = {
        "num_classes": 2,
        "inc": 3,
        "dropout": 0.2,
        "model_type": "vit_b_16",
        "pretrained": True,
        "vit_resolution": 256,
    }
    model = VisionTransformer(vit_config)
    model.eval()

    input_image = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(input_image)
    print(output.shape) # torch.Size([1, 2])