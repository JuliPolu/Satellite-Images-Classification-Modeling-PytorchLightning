import torch
import torch.nn as nn
from torch import Tensor
from typing import Union


class FocalLoss(nn.Module):
    def __init__(
            self,
            gamma: float = 1,
            weights: Union[None, Tensor] = torch.tensor([119.4071, 119.0559,  15.0089, 193.6794,   1.4238,   5.5749, 413.0510, 46.9594,   5.4620,  11.0598,   5.0154, 404.7900,   9.0415,   1.0791, 3.2870, 121.9247,  19.3772]),
            reduction: str = 'mean',
            eps: float = 1e-16
            ) -> None:
        super().__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError(f'Reduction {reduction} not implemented.')
        
        self.gamma = gamma
        self.weights = weights.to('cuda:0')
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        # Apply sigmoid activation for multi-label problem
        probs = torch.sigmoid(logits)

        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1.0 - pt).pow(self.gamma)
        loss = -focal_weight * torch.log(pt + self.eps)

        if self.weights is not None:
            loss = loss * self.weights.view(1, -1)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass

        return loss
