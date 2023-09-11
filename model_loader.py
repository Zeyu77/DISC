import torch
import torch.nn as nn
import torchvision.models as models
import os
from torch.nn import Parameter

import torch.nn.init as init
import torch.nn.functional as F

def load_model(arch, code_length):
    """
    Load CNN model.

    Args
        arch(str): Model name.
        code_length(int): Hash code length.

    Returns
        model(torch.nn.Module): CNN model.
    """

    if arch == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Identity()
        model = ModelWrapper(model, 2048, code_length)
    else:
        raise ValueError("Invalid model name!")

    return model



class ModelWrapper(nn.Module):
    """
    Add tanh activate function into model.

    Args
        model(torch.nn.Module): CNN model.
        last_node(int): Last layer outputs size.
        code_length(int): Hash code length.
    """
    def __init__(self, model, last_node, code_length):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.code_length = code_length
        self.hash_layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(last_node, code_length),
            nn.Tanh(),
        )

        # Extract features
        self.extract_features = False
        # Add hash_layer_source to generate hash codes for simulated source features
        self.hash_layer_source = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(last_node, code_length),
            nn.Tanh(),
        )
        


    def forward(self, x):
        if self.extract_features:
            feature = self.model(x)
            return feature, self.hash_layer(feature)
        else:
            feature = self.model(x)
            hash_code = self.hash_layer(feature)

            return feature, hash_code


    def set_extract_features(self, flag):
        """
        Extract features.

        Args
            flag(bool): true, if one needs extract features.
        """
        self.extract_features = flag


