
import torch
from torch import nn
from typing import List
from src.config import LossConfig


# Focal Loss for Multi-label Classification
class FocalLossMultiLabel(nn.Module):
    def __init__(self, gamma):
        super(FocalLossMultiLabel, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')
        loss = bce_with_logits(input, target)
        pt = torch.exp(-loss)
        loss = (1 - pt)** self.gamma * loss
        return loss.mean()
