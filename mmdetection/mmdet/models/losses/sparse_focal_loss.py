# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss

from ..builder import LOSSES
from .utils import weight_reduce_loss


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
    pred_sigmoid = pred.sigmoid() 
    target = target.type_as(pred)
    
    # Calculate px (1 - pt)
    px = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target) 

    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * px.pow(gamma) 
    
    # Original focal loss calculation
    original_focal_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight   
    
    hard_negatives = ((1 - px) < thresh) & (target == 0) 
    
    # Apply positive and hard negative weights
    loss = original_focal_loss.clone()
    loss = torch.where(target == 1, loss * positive_weight, loss) 
    loss = torch.where(hard_negatives, loss * hard_negative_weight, loss)

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
class SparseFocalLoss(nn.Module):
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
        super(SparseFocalLoss, self).__init__()
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
                gamma=self.gamma,  # 2
                alpha=self.alpha,  # 0.25
                thresh=self.thresh,  # 1
                reduction=reduction, 
                avg_factor=avg_factor,
                hard_negative_weight=self.hard_negative_weight,  # 0.4
                positive_weight=self.positive_weight  # 1.0
            )
        else:
            raise NotImplementedError("Only sigmoid focal loss is implemented.")
        
        return loss_cls

