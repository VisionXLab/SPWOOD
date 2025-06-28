# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss

from ..builder import LOSSES
from .utils import weight_reduce_loss


def py_mirror_sigmoid_focal_loss(pred,
                                 target,
                                 weight=None,
                                 gamma_1=2.0,
                                 gamma_2=2.0,
                                 alpha=0.25,
                                 thresh=0.5,
                                 beta=1.0,
                                 reduction='mean',
                                 avg_factor=None):
    """Mirror Focal Loss (BRL) with binary cross entropy for hard negatives."""
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    
    # 计算原始的 Focal Loss
    px = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target) # px = 1- pt
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * px.pow(gamma_1)
    original_focal_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
    
    # pt < thr 表示为 hard negetive，使用 Mirror Focal Loss 计算
    hard_negatives = ((1 - px) < thresh) & (target == 0)
    hard_neg_weight = alpha * (1 - px).pow(gamma_2)
    # Calculate the hard negative term
    hard_neg_term = F.binary_cross_entropy_with_logits(pred, 1 - target, reduction='none') * hard_neg_weight * beta
    
    # Initialize loss as original focal loss and then mask replace with hard neg term
    loss = original_focal_loss.clone()
    loss = torch.where(hard_negatives, hard_neg_term, loss)

    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class MirrorFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma_1=2.0,
                 gamma_2=2.0,
                 alpha=0.25,
                 thresh=0.5,
                 beta=1.0,
                 reduction='mean',
                 loss_weight=1.0):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            activated (bool, optional): Whether the input is activated.
                If True, it means the input has been activated and can be
                treated as probabilities. Else, it should be treated as logits.
                Defaults to False.
        """
        super(MirrorFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma1 = gamma_1
        self.gamma2 = gamma_2
        self.alpha = alpha
        self.thresh = thresh
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            target = target.long()
            num_classes = pred.size(1)
            target = F.one_hot(target, num_classes=num_classes + 1)
            target = target[:, :num_classes]
            calculate_loss_func = py_mirror_sigmoid_focal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                weight,
                gamma_1=self.gamma1,
                gamma_2=self.gamma2,
                alpha=self.alpha,
                thresh=self.thresh,
                beta=self.beta,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls