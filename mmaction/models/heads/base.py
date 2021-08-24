# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import mmit_mean_average_precision, top_k_accuracy
from ..builder import build_loss


class AvgConsensus(nn.Module):
    """Average consensus module.

    Args:
        dim (int): Decide which dim consensus function to apply.
            Default: 1.
    """

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """Defines the computation performed at every call."""
        return x.mean(dim=self.dim, keepdim=True)


class NeXtVLAD(nn.Module):
    """This is a PyTorch implementation of the NeXtVLAD + Context Gating model.

    For more
    information, please refer to the paper,
    https://static.googleusercontent.com/media/research.google.com/zh-CN//youtube8m/workshop2018/p_c03.pdf
    """

    def __init__(self,
                 feature_size=1024,
                 cluster_size=128,
                 expansion=2,
                 groups=8,
                 hidden_size=2048,
                 gating_reduction=8,
                 dropout=0.4):
        super().__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.expansion = expansion
        self.groups = groups
        self.hidden_size = hidden_size
        self.group_feature_size = feature_size * expansion // groups
        self.vlad_size = self.group_feature_size * cluster_size

        # NeXtVLAD
        self.cluster_weights = nn.Parameter(
            torch.empty(expansion * feature_size, groups * cluster_size))
        self.cluster_weights2 = nn.Parameter(
            torch.empty(1, self.group_feature_size, cluster_size))
        nn.init.kaiming_normal_(self.cluster_weights)
        nn.init.kaiming_normal_(self.cluster_weights2)

        self.fc_expansion = nn.Linear(feature_size, expansion * feature_size)
        self.fc_group_attention = nn.Sequential(
            nn.Linear(expansion * feature_size, groups), nn.Sigmoid())
        self.activation_bn = nn.BatchNorm1d(groups * cluster_size)
        self.cluster_softmax = nn.Softmax(dim=-1)
        self.vlad_bn = nn.BatchNorm1d(self.group_feature_size * cluster_size)

        # Context Gating
        self.cg_proj = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.vlad_size, hidden_size),
            nn.BatchNorm1d(hidden_size))
        self.gate1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // gating_reduction),
            nn.BatchNorm1d(hidden_size // gating_reduction), nn.ReLU())
        self.gate2 = nn.Sequential(
            nn.Linear(hidden_size // gating_reduction, hidden_size),
            nn.Sigmoid())

    def forward(self, input):
        """
        Args:
            input: (torch.tensor) (B, num_segs, feature_size, 1, 1)
            Output: (torch.tensor) (B, 1, hidden_size, 1, 1)
        """
        B, num_segs = input.shape[:2]
        input = input.view(B * num_segs, self.feature_size)
        # [B*num_segs, feature_size]

        # NeXtVLAD
        input = self.fc_expansion(input)
        # [B, num_segs, expansion*feature_size]
        attention = self.fc_group_attention(input)
        attention = attention.reshape(-1, num_segs * self.groups)
        # [B, num_segs*groups]
        reshaped_input = input.reshape(-1, self.expansion * self.feature_size)
        # [B*num_segs, expansion*feature_size]
        activation = self.activation_bn(reshaped_input @ self.cluster_weights)
        # [B*num_segs, groups*cluster_size]
        activation = activation.reshape(-1, num_segs * self.groups,
                                        self.cluster_size)
        # [B, num_segs*groups, cluster_size]
        activation = self.cluster_softmax(activation)
        activation = activation * attention.unsqueeze(-1)
        # [B, num_segs*groups, cluster_size]
        a_sum = activation.sum(dim=1, keepdim=True)
        # [B, 1, cluster_size]
        a = a_sum * self.cluster_weights2
        # [B, group_feature_size, cluster_size]
        activation = activation.transpose(1, 2)
        # [B, cluster_size, num_segs*groups]
        input = input.reshape(-1, num_segs * self.groups,
                              self.group_feature_size)
        # [B, num_segs*groups, group_feature_size]
        vlad = torch.matmul(activation, input)
        # [B, cluster_size, group_feature_size]
        vlad = vlad.transpose(1, 2)
        # [B, group_feature_size, cluster_size]
        vlad = vlad - a
        vlad = F.normalize(vlad, dim=1)
        # [B, group_feature_size, cluster_size]
        vlad = vlad.reshape(-1, self.cluster_size * self.group_feature_size)
        vlad = self.vlad_bn(vlad)
        # [B, group_feature_size*cluster_size]

        # Context Gating
        activation = self.cg_proj(vlad)
        # [B, hidden_size]
        gates = self.gate1(activation)
        # [B, hidden_size/gating_reduction]
        gates = self.gate2(gates)
        # [B, hidden_size]
        vlad_cg = activation * gates
        # [B, hidden_size]

        return vlad_cg.view(B, 1, self.hidden_size, 1, 1)


class BaseHead(nn.Module, metaclass=ABCMeta):
    """Base class for head.

    All Head should subclass it.
    All subclass should overwrite:
    - Methods:``init_weights``, initializing weights in some modules.
    - Methods:``forward``, supporting to forward both for training and testing.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss', loss_weight=1.0).
        multi_class (bool): Determines whether it is a multi-class
            recognition task. Default: False.
        label_smooth_eps (float): Epsilon used in label smooth.
            Reference: arxiv.org/abs/1906.02629. Default: 0.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 multi_class=False,
                 label_smooth_eps=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.loss_cls = build_loss(loss_cls)
        self.multi_class = multi_class
        self.label_smooth_eps = label_smooth_eps

    @abstractmethod
    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""

    @abstractmethod
    def forward(self, x):
        """Defines the computation performed at every call."""

    def loss(self, cls_score, labels, **kwargs):
        """Calculate the loss given output ``cls_score``, target ``labels``.

        Args:
            cls_score (torch.Tensor): The output of the model.
            labels (torch.Tensor): The target output of the model.

        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'top1_acc', 'top5_acc'(optional).
        """
        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_socre` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if not self.multi_class and cls_score.size() != labels.size():
            top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(), (1, 5))
            losses['top1_acc'] = torch.tensor(
                top_k_acc[0], device=cls_score.device)
            losses['top5_acc'] = torch.tensor(
                top_k_acc[1], device=cls_score.device)

        elif self.multi_class:
            mAP = mmit_mean_average_precision(cls_score.detach().cpu().numpy(),
                                              labels.detach().cpu().numpy())
            losses['mAP'] = torch.tensor(mAP, device=cls_score.device)
            if self.label_smooth_eps != 0:
                labels = ((1 - self.label_smooth_eps) * labels +
                          self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(cls_score, labels, **kwargs)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls

        return losses
