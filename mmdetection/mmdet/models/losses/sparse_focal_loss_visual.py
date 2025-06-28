# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss

from ..builder import LOSSES
from .utils import weight_reduce_loss
import os
import numpy as np
import cv2

def py_sparse_sigmoid_focal_loss(pred,
                                 target,
                                 weight=None,
                                 gamma=2.0,
                                 alpha=0.25,
                                 thresh=0.5,
                                 reduction='mean',
                                 avg_factor=None,
                                 hard_negative_weight=0.3,
                                 positive_weight=1.0,
                                 iter=0):
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
    
    # Calculate px (1 - pt), where pt is the probability of correct classification
    px = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target) # px = 1 - pt

    # #------------------------------Statistic Start---------------------------------------
    # if iter % 3200 == 0:
    #     max_lines = 100000
    #     save_dir = '/workspace/animax/MCL/tools/visualization/'  # 指定保存目录
    #     os.makedirs(save_dir, exist_ok=True)  # 确保目录存在

    #     # 动态生成文件名，根据当前 iter 值
    #     stat_file = os.path.join(save_dir, f"px_distribution_iter_{iter}.txt")

    #     # 检查文件是否已经存在，并统计当前行数
    #     current_lines = 0
    #     try:
    #         with open(stat_file, 'r') as f:
    #             current_lines = sum(1 for _ in f)
    #     except FileNotFoundError:
    #         pass

    #     # 如果未达到最大行数，继续写入
    #     if current_lines < max_lines:
    #         with open(stat_file, 'a') as f:
    #             px_values = px.detach().cpu().numpy().flatten()
    #             for value in px_values:
    #                 if current_lines >= max_lines:
    #                     break
    #                 f.write(f"{value}\n")
    #                 current_lines += 1
    #     print(f'sparse focal loss px save to: {stat_file}')
    # #------------------------------Statistic End---------------------------------------



    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * px.pow(gamma)
    
    # Original focal loss calculation
    original_focal_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
    
    # Identify hard negatives (samples with pt < thresh) 这里等同于 p > 1 - thr p表示分类为正样本的概率
    hard_negatives = ((1 - px) < thresh) & (target == 0)
    
    # Apply positive and hard negative weights
    loss = original_focal_loss.clone()

    #------------------------------Statistic Start---------------------------------------
    if (iter - 1) % 3200 == 0:
        save_dir = '/workspace/animax/MCL/tools/visualization/'  # 保存目录
        os.makedirs(save_dir, exist_ok=True)  # 确保目录存在

        # 动态生成文件名
        stat_file = os.path.join(save_dir, f"px_loss_contribution_iter_{iter}.txt")

        # px 值和 loss 值
        px_values = px.detach().cpu().numpy().flatten()  # px 值 (分类为正样本的概率)
        loss_values = loss.detach().cpu().numpy().flatten()  # 对应的 loss 值
        target_values = target.detach().cpu().numpy().flatten()  # 目标值 (0 或 1)

        # 定义 100 个区间：[0.0~0.01], [0.01~0.02], ..., [0.99~1.0]
        bins = np.linspace(0, 1.0, 101)  # 100 个区间的边界
        bin_names = [f"{bins[i]:.2f}~{bins[i+1]:.2f}" for i in range(len(bins) - 1)]  # 区间名称
        bin_loss_sum_negative = [0] * 100  # 每个区间的负样本 loss 加和
        bin_loss_sum_positive = [0] * 100  # 每个区间的正样本 loss 加和
        total_loss_negative = 0  # 总负样本 loss
        total_loss_positive = 0  # 总正样本 loss

        # 筛选负样本和正样本，按区间进行分类
        for i in range(len(px_values)):
            px_val = px_values[i]
            if target_values[i] == 0:  # 负样本
                for j in range(len(bins) - 1):
                    if bins[j] <= px_val < bins[j + 1]:
                        bin_loss_sum_negative[j] += loss_values[i]
                        total_loss_negative += loss_values[i]
                        break
            elif target_values[i] == 1:  # 正样本
                for j in range(len(bins) - 1):
                    if bins[j] <= px_val < bins[j + 1]:
                        bin_loss_sum_positive[j] += loss_values[i]
                        total_loss_positive += loss_values[i]
                        break

        # 将结果保存到文本文件
        with open(stat_file, 'w') as f:
            f.write("Loss Contribution in Each px Range\n")
            f.write(f"{'Range':<16}{'Negative Loss':<20}{'Positive Loss':<20}\n")
            f.write("-" * 56 + "\n")
            
            for name, neg_loss, pos_loss in zip(bin_names, bin_loss_sum_negative, bin_loss_sum_positive):
                f.write(f"{name:<16}{neg_loss:<20.4f}{pos_loss:<20.4f}\n")
            
            # 总损失
            f.write("\n")
            f.write(f"{'Total Negative Loss':<30}{total_loss_negative:<20.4f}\n")
            f.write(f"{'Total Positive Loss':<30}{total_loss_positive:<20.4f}\n")
            f.write(f"{'Total Loss':<30}{total_loss_negative + total_loss_positive:<20.4f}\n")
        
        print(f'px loss contribution saved to: {stat_file}')
    #------------------------------Statistic End---------------------------------------

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
class SparseFocalLossVisual(nn.Module):
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
        super(SparseFocalLossVisual, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss is supported currently.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.thresh = thresh
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.hard_negative_weight = hard_negative_weight
        self.positive_weight = positive_weight
        self.iter = 0

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
                positive_weight=self.positive_weight,
                iter=self.iter
            )
        else:
            raise NotImplementedError("Only sigmoid focal loss is implemented.")
        self.iter += 1
        return loss_cls

