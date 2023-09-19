import torch
import torch.nn as nn
from timm import create_model



def initialize_model(num_classes, freeze_grad, unfreeze_num, **model_kwargs):

    model = create_model(num_classes=num_classes, **model_kwargs)

    if freeze_grad:
        for param in model.parameters():
            param.requires_grad = False

        for param in list(model.parameters())[-unfreeze_num:]:
            param.requires_grad = True

        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)