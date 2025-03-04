'''
Convolutional Vision Transformer (CvT)
reference:
- https://paperswithcode.com/paper/cvt-introducing-convolutions-to-vision
- https://huggingface.co/microsoft/cvt-21
'''

from transformers import CvtForImageClassification
import torch
import torch.nn as nn
import os
from metrics.registry import BACKBONE


@BACKBONE.register_module(module_name="cvt")
class CvT(nn.Module):
    def __init__(self, config):
        super(CvT, self).__init__()
        """ Constructor
        Args:
            config: configuration file with the dict format
        """
        self.num_classes = config["num_classes"]
        self.pretrained_path = config.get("pretrained_path", "")
        self.from_pretrained = config["from_pretrained"]

        # Load the Convolutional Vision Transformer (CvT) model
        #self.cvt = CvtForImageClassification.from_pretrained('microsoft/cvt-21')
        self.cvt = CvtForImageClassification.from_pretrained(self.from_pretrained)
        #output self.cvt to txt
        # with open('cvt.txt', 'w') as f:
        #    f.write(str(self.cvt))
        self.cvt.classifier = nn.Identity()

        # Initialize the last_layer layer
        self.last_layer = nn.Linear(384, self.num_classes)
        if "w24" in self.from_pretrained:
            self.last_layer = nn.Linear(1024, self.num_classes)

    def features(self, x):
        x = self.cvt(x).logits
        return x

    def classifier(self, x):
        x = self.last_layer(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    config = {
        "num_classes": 2
    }
    model = CvT(config)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)

    print(model)
    print(y)
    print(y.size())