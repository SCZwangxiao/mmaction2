# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import trunc_normal_init

from ...core import mean_average_precision, mmit_mean_average_precision
from ..builder import HEADS
from .base import BaseHead


@HEADS.register_module()
class VanillaHead(BaseHead):
    """Vanilla classification head.

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
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout_ratio=0.4,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.dropout_ratio = dropout_ratio
        self.num_verticals = num_verticals

        # if self.num_verticals:
        #     self.vertical_embed = nn.Embedding(num_verticals, self.in_channels)
        # else:
        #     self.vertical_embed = None

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        # self.finetune_fc = nn.Sequential(
        #     nn.Linear(self.in_channels, self.in_channels),
        #     nn.BatchNorm1d(self.in_channels)
        # )
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def forward(self, x, vertical, num_segs):
        # [N, num_clips, in_channels]
        x = x.mean(1)
        # [N, in_channels]
        # x = self.finetune_fc(x)
        # if self.vertical_embed:
        #     x = x + self.vertical_embed(vertical)
        if self.dropout is not None:
            x = self.dropout(x)
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score

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
