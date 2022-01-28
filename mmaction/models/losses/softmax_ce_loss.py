# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from ..builder import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register_module()
class SoftmaxCELoss(BaseWeightedLoss):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self,  
                 loss_weight=1.0):
        super().__init__(loss_weight=loss_weight)

    def _forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        targets = y
        anti_targets = 1 - y

        # Calculating exp
        e_pos = targets * torch.exp(-x)
        e_neg = anti_targets * torch.exp(x)

        # Logsumexp
        loss_pos = torch.log(1 + e_pos.sum(1))
        loss_neg = torch.log(1 + e_neg.sum(1))

        loss = loss_pos + loss_neg
        return loss