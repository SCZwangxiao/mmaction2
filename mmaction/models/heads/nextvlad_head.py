# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import trunc_normal_init

from ...core import mean_average_precision, mmit_mean_average_precision
from ..builder import HEADS
from .base import BaseHead


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
        self.gating_reduction = gating_reduction
        self.group_feature_size = feature_size * expansion // groups
        self.vlad_size = self.group_feature_size * cluster_size

        # NeXtVLAD
        self.cluster_weights = nn.Parameter(
            torch.empty(expansion * feature_size, groups * cluster_size))
        self.c = nn.Parameter(
            torch.empty(1, self.group_feature_size, cluster_size))

        self.fc_expansion = nn.Linear(feature_size, expansion * feature_size)
        self.fc_group_attention = nn.Sequential(
            nn.Linear(expansion * feature_size, groups), nn.Sigmoid())
        self.activation_bn = nn.BatchNorm1d(groups * cluster_size)
        self.vlad_bn = nn.BatchNorm1d(self.group_feature_size * cluster_size)

        # Context Gating
        self.cg_proj = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.vlad_size, hidden_size),
            nn.BatchNorm1d(hidden_size))
        if gating_reduction:
            self.gate1 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // gating_reduction),
                nn.BatchNorm1d(hidden_size // gating_reduction), nn.ReLU())
            self.gate2 = nn.Sequential(
                nn.Linear(hidden_size // gating_reduction, hidden_size),
                nn.Sigmoid())

    def init_weights(self):
        nn.init.kaiming_normal_(self.cluster_weights)
        nn.init.kaiming_normal_(self.c)

    def forward(self, input):
        """
        Args:
            input: (torch.tensor) (B, num_segs, feature_size)
            Output: (torch.tensor) (B, hidden_size)
        """
        B, num_segs = input.shape[:2]
        # NeXtVLAD
        input = self.fc_expansion(input)
        # [B, num_segs, expansion*feature_size]
        group_attention = self.fc_group_attention(input)
        group_attention = group_attention.reshape(-1, num_segs * self.groups)
        # [B, num_segs*groups]
        reshaped_input = input.reshape(-1, self.expansion * self.feature_size)
        # [B*num_segs, expansion*feature_size]
        activation = self.activation_bn(reshaped_input @ self.cluster_weights)
        # [B*num_segs, groups*cluster_size]
        activation = activation.reshape(-1, num_segs * self.groups,
                                        self.cluster_size)
        # [B, num_segs*groups, cluster_size]
        activation = F.softmax(activation, dim=-1)
        activation = activation * group_attention.unsqueeze(-1)
        # [B, num_segs*groups, cluster_size]
        a_sum = activation.sum(dim=1, keepdim=True)
        # [B, 1, cluster_size]
        a = a_sum * self.c
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
        if self.gating_reduction:
            gates = self.gate1(activation)
            # [B, hidden_size/gating_reduction]
            gates = self.gate2(gates)
            # [B, hidden_size]
            vlad_cg = activation * gates
            # [B, hidden_size]

        return vlad_cg


@HEADS.register_module()
class NextVLADHead(BaseHead):
    """NextVLAD classification head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Defaults to `dict(type='CrossEntropyLoss')`.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_verticals=None,
                 online_model=False,
                 cluster_size=128,
                 expansion=2,
                 groups=8,
                 hidden_size=2048,
                 gating_reduction=8,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout_ratio=0.4,
                 use_tde=False,
                 tau=16,
                 alpha=3,
                 gamma=1 / 32,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.dropout_ratio = dropout_ratio
        self.num_verticals = num_verticals
        self.online_model = online_model
        self.hidden_size = hidden_size
        self.use_tde = use_tde
        self.tau = tau
        self.alpha = alpha
        self.gamma = gamma

        if self.online_model:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.next_vlad = NeXtVLAD(
            feature_size=in_channels,
            cluster_size=cluster_size,
            expansion=2,
            groups=8,
            hidden_size=hidden_size,
            gating_reduction=8,
            dropout=dropout_ratio)
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        if self.use_tde:
            self.register_buffer(
                'x_mean', torch.zeros((1, hidden_size), requires_grad=False))
        self.fc_cls = nn.Linear(hidden_size, self.num_classes, bias=False)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        self.next_vlad.init_weights()

    def cos_sim(self, x_normed, x_mean):
        """
        Args:
            x_normed (N, hidden_size)
            x_mean (1, hidden_size)
        Return:
            sim (N, 1)
        """
        sim = x_normed @ x_mean.t()
        # [N, 1]
        x_mean = x_mean.squeeze(0)
        sim = sim / torch.norm(
            x_normed, dim=1, keepdim=True) / torch.norm(x_mean)
        return sim

    def forward(self, x, num_segs, vertical=None):
        if self.online_model:
            return self.forward_online(x, num_segs)
        else:
            return self.forward_offline(x, num_segs, vertical)

    def forward_offline(self, x, num_segs, vertical):
        # [N, num_segs, in_channels]
        x = self.next_vlad(x)
        # [N, hidden_size]
        if self.use_tde:
            self.x_mean = 0.9 * self.x_mean + x.detach().mean(0, keepdim=True)
        x_normed = F.normalize(x, dim=1)
        # [N, hidden_size]
        if self.use_tde and not self.training:
            x_normed = x_normed - self.alpha * self.cos_sim(
                x_normed, self.x_mean) * F.normalize(
                    self.x_mean, dim=1)
        if self.dropout is not None:
            x_normed = self.dropout(x_normed)
        # [N, num_classes]
        classifier_norm = torch.norm(self.fc_cls.weight, 2,
                                     1).unsqueeze(0) + self.gamma
        # [1, hidden_size]
        cls_score = self.tau * self.fc_cls(x_normed) / classifier_norm
        # [N, num_classes]
        return cls_score

    def forward_online(self, x, num_segs):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
            num_segs (int): Number of segments into which a video
                is divided.
        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N * num_segs, in_channels, 7, 7]
        if self.avg_pool is not None:
            if isinstance(x, tuple):
                shapes = [y.shape for y in x]
                assert 1 == 0, f'x is tuple {shapes}'
            x = self.avg_pool(x)
            # [N * num_segs, in_channels, 1, 1]
        x = x.reshape((-1, num_segs) + x.shape[1:]).squeeze(3).squeeze(3)
        # [N, num_segs, in_channels]
        return self.forward_offline(x, num_segs, None)

    def build_labels(self, cls_score, sparse_labels):
        batch_size = len(sparse_labels)
        labels = torch.zeros_like(cls_score)
        for b, sparse_label in enumerate(sparse_labels):
            labels[b, sparse_label] = 1.
        return labels

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

        labels = self.build_labels(cls_score, labels)

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
            cls_score_cpu = cls_score.detach().cpu().numpy()
            labels_cpu = labels.detach().cpu().numpy()
            mAP_sample = mmit_mean_average_precision(cls_score_cpu, labels_cpu)
            losses['mAP_sample'] = torch.tensor(
                mAP_sample, device=cls_score.device)
            mAP_label = mean_average_precision(cls_score_cpu, labels_cpu)
            losses['mAP_label'] = torch.tensor(
                mAP_label, device=cls_score.device)
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
