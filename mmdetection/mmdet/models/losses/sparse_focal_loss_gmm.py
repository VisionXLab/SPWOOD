# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss

from ..builder import LOSSES
from .utils import weight_reduce_loss

import sklearn.mixture as skm


def gmm_policy(self, scores, given_gt_thr=0.02, policy='high'):
    """The policy of choosing pseudo label.

    The previous GMM-B policy is used as default.
    1. Use the predicted bbox to fit a GMM with 2 center.
    2. Find the predicted bbox belonging to the positive
        cluster with highest GMM probability.
    3. Take the class score of the finded bbox as gt_thr.

    Args:
        scores (nd.array): The scores.

    Returns:
        float: Found gt_thr.

    """
    if len(scores) < 4:
        return given_gt_thr
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if len(scores.shape) == 1:
        scores = scores[:, np.newaxis]
    means_init = [[np.min(scores)], [np.max(scores)]]
    weights_init = [1 / 2, 1 / 2]
    precisions_init = [[[1.0]], [[1.0]]]
    gmm = skm.GaussianMixture(
        2,
        weights_init=weights_init,
        means_init=means_init,
        precisions_init=precisions_init)
    gmm.fit(scores)
    gmm_assignment = gmm.predict(scores)
    gmm_scores = gmm.score_samples(scores)
    assert policy in ['middle', 'high']
    if policy == 'high':
        if (gmm_assignment == 1).any():
            # 将所有低分簇样本的GMM概率设为负无穷，确保它们不被选中
            gmm_scores[gmm_assignment == 0] = -np.inf
            # 找到GMM概率最高的那个点的索引
            indx = np.argmax(gmm_scores, axis=0)
            # 筛选出同时满足以下两个条件的点：
            # 1. 属于高分簇 (gmm_assignment == 1)
            # 2. 得分不低于GMM概率最高那个点的得分 (scores >= scores[indx])
            pos_indx = (gmm_assignment == 1) & (
                scores >= scores[indx]).squeeze()
            pos_thr = float(scores[pos_indx].min())
            # pos_thr = max(given_gt_thr, pos_thr)
        else:
            pos_thr = given_gt_thr
    elif policy == 'middle':
        if (gmm_assignment == 1).any():
            # 直接取所有被分到高分簇的点的最低分作为阈值
            pos_thr = float(scores[gmm_assignment == 1].min())
            # pos_thr = max(given_gt_thr, pos_thr)
        else:
            pos_thr = given_gt_thr

    return pos_thr


def py_sparse_sigmoid_focal_loss(pred,
                                 target,
                                 weight=None,
                                 gamma=2.0,
                                 alpha=0.25,
                                 thresh=0.5,
                                 reduction='mean',
                                 avg_factor=None,
                                 hard_negative_weight=0.3,
                                 positive_weight=1.0):
    """Sparse Sigmoid Focal Loss with hard negative and positive weighting.
    
    Args:
        pred (torch.Tensor): The predictions (logits) from the model.
        target (torch.Tensor): Ground truth labels (0 or 1).
        weight (torch.Tensor, optional): Loss weight for each prediction.
        gamma (float): Focusing parameter for modulating factor. Default: 2.0.
        alpha (float): Balancing parameter for Focal Loss. Default: 0.25.
        thresh (float): Threshold for classifying hard negatives. Default: 0.5.
        reduction (str): Method to reduce the loss. Options: "none", "mean", "sum".
        avg_factor (int, optional): Average factor for loss scaling.
        hard_negative_weight (float): Weight for hard negative samples. Default: 0.3.
        positive_weight (float): Weight for positive samples. Default: 1.0.

    Returns:
        torch.Tensor: Calculated loss value.
    """
    pred_sigmoid = pred.sigmoid()  # 计算概率
    target = target.type_as(pred)
    
    # Calculate px (1 - pt), where pt is the probability of correct classification 这里的px恒等于正确类别（可能是0、可能是1）的概率
    px = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target) # px = 1 - pt

    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * px.pow(gamma)
    
    # Original focal loss calculation
    original_focal_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
    
    # Identify hard negatives (samples with pt < thresh) 这里等同于 p > 1 - thr p表示分类为正样本的概率
    hard_negatives = ((1 - px) < thresh) & (target == 0)  # 难分负样本,是target为0,而且预测值小于阈值的
    
    # Apply positive and hard negative weights
    loss = original_focal_loss.clone()
    loss = torch.where(target == 1, loss * positive_weight, loss)  # Positive sample weighting
    loss = torch.where(hard_negatives, loss * hard_negative_weight, loss)  # Hard negative weighting

    if weight is not None:
        # Adjust weight shape if necessary to match the loss dimensions
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    # Reduce loss with specified reduction method
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class SparseFocalLossGMM(nn.Module):
    """Implementation of Sparse Focal Loss with support for hard negative and positive weighting.
    
    Args:
        use_sigmoid (bool): Use sigmoid for predictions. Only supported option is True.
        gamma (float): Modulating factor gamma for Focal Loss. Default: 2.0.
        alpha (float): Balancing factor alpha for Focal Loss. Default: 0.25.
        thresh (float): Threshold for hard negative classification. Default: 0.5.
        reduction (str): Reduction method for loss. Options: "none", "mean", "sum".
        loss_weight (float): Overall weight for the loss. Default: 1.0.
        hard_negative_weight (float): Weight applied to hard negatives. Default: 0.3.
        positive_weight (float): Weight applied to positive samples. Default: 1.0.
    """
    
    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 thresh=0.5,
                 reduction='mean',
                 loss_weight=1.0,
                 hard_negative_weight=0.3,
                 positive_weight=1.0):
        super(SparseFocalLossGMM, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss is supported currently.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.thresh = thresh
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.hard_negative_weight = hard_negative_weight
        self.positive_weight = positive_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Compute Sparse Focal Loss with optional hard negative and positive weighting.
        
        Args:
            pred (torch.Tensor): The predictions from the model.
            target (torch.Tensor): Ground truth labels for predictions.
            weight (torch.Tensor, optional): Loss weight for each prediction.
            avg_factor (int, optional): Average factor for scaling the loss.
            reduction_override (str, optional): Override the default reduction method.
            
        Returns:
            torch.Tensor: Computed loss value.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        
        if self.use_sigmoid:
            target = target.long()
            num_classes = pred.size(1)
            target = F.one_hot(target, num_classes=num_classes + 1)
            target = target[:, :num_classes]
            
            loss_cls = self.loss_weight * py_sparse_sigmoid_focal_loss(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                thresh=self.thresh,
                reduction=reduction,
                avg_factor=avg_factor,
                hard_negative_weight=self.hard_negative_weight,
                positive_weight=self.positive_weight
            )
        else:
            raise NotImplementedError("Only sigmoid focal loss is implemented.")
        
        return loss_cls

