#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/18 21:01
# @Author : WeiHua
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmrotate.models import ROTATED_LOSSES, build_loss
from mmrotate.core import build_bbox_coder
from mmdet.core.anchor.point_generator import MlvlPointGenerator
import numpy as np
from mmrotate.core import poly2obb_np
import cv2
import mmcv

INF = 1e8
CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
            'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
            'basketball-court', 'storage-tank', 'soccer-ball-field',
            'roundabout', 'harbor', 'swimming-pool', 'helicopter')


@ROTATED_LOSSES.register_module()
class RotatedDTLossAssignerAssistentV3forLabeledData(nn.Module):
    def __init__(self, cls_channels=len(CLASSES), loss_type='origin', bbox_loss_type='l1', image_class_prompt_path='/workspace/animax/MCL/tools/Assinger_Assistent/image_class_prompt_from_chat_with_percent5_label_modified.pt'):
        super(RotatedDTLossAssignerAssistentV3forLabeledData, self).__init__()
        self.cls_channels = cls_channels
        assert bbox_loss_type in ['l1', 'iou']
        self.bbox_loss_type = bbox_loss_type
        self.bbox_coder = build_bbox_coder(dict(type='DistanceAnglePointCoder', angle_version='le90'))
        self.prior_generator = MlvlPointGenerator([8, 16, 32, 64, 128])
        if self.bbox_loss_type == 'l1':
            self.bbox_loss = nn.SmoothL1Loss(reduction='none')
        else:
            self.bbox_coder = build_bbox_coder(dict(type='DistanceAnglePointCoder', angle_version='le90'))
            self.prior_generator = MlvlPointGenerator([8, 16, 32, 64, 128])
            self.bbox_loss = build_loss(dict(type='RotatedIoULoss', reduction='none'))
        self.loss_type = loss_type

        self.image_class_prompt_path = image_class_prompt_path
        self.image_class_prompt = torch.load(self.image_class_prompt_path)
                # 预处理类别映射，将类别名转换为索引
        self.class_name_to_index = {
            filename: torch.tensor(
                [CLASSES.index(cls) for cls in exist_classes if cls in CLASSES],
                device='cpu'  # 初始化时为CPU，运行时转移到设备
            )
            for filename, exist_classes in self.image_class_prompt.items()
        }

    def convert_shape(self, logits):
        cls_scores, bbox_preds, angle_preds, centernesses = logits
        assert len(cls_scores) == len(bbox_preds) == len(angle_preds) == len(centernesses)

        batch_size = cls_scores[0].shape[0]   
        cls_scores = torch.cat([
            x.permute(0, 2, 3, 1).reshape(batch_size, -1, self.cls_channels) for x in cls_scores
        ], dim=1).view(-1, self.cls_channels)
        bbox_preds = torch.cat([
            torch.cat([x, y], dim=1).permute(0, 2, 3, 1).reshape(batch_size, -1, 5) for x, y in
            zip(bbox_preds, angle_preds)
        ], dim=1).view(-1, 5)
        centernesses = torch.cat([
            x.permute(0, 2, 3, 1).reshape(batch_size, -1, 1) for x in centernesses
        ], dim=1).view(-1, 1)
        return cls_scores, bbox_preds, centernesses

    def forward(self, teacher_logits, student_logits, ratio=0.03, img_metas=None, bbox_head=None, **kwargs):
        # (21824, 15)  (21824, 5)  (21824, 1)
        t_cls_scores, t_bbox_preds, t_centernesses = self.convert_shape(teacher_logits)
        s_cls_scores, s_bbox_preds, s_centernesses = self.convert_shape(student_logits)
        gt_bboxes = img_metas['gt_bboxes']
        gt_labels = img_metas['gt_labels']

        all_level_points = self.prior_generator.grid_priors(
            [featmap.size()[-2:] for featmap in teacher_logits[0]],
            dtype=s_bbox_preds.dtype,
            device=s_bbox_preds.device)

        batch_gt_instances = []
        batch_gt_instance = {}
        for i in range(len(img_metas['img'])):
            batch_gt_instance['bboxes'] = img_metas['gt_bboxes'][i]
            batch_gt_instance['labels'] = img_metas['gt_labels'][i]
            batch_gt_instance['bids'] = torch.randn(1, 4).to('cuda')
            batch_gt_instances.append(batch_gt_instance)
        if not batch_gt_instances:
            batch_gt_instance['bboxes'] = torch.empty(0, 5).to('cuda')
            batch_gt_instance['labels'] = torch.empty(0, 2).to('cuda')
            batch_gt_instance['bids'] = torch.empty(0, 4).to('cuda')
            batch_gt_instances.append(batch_gt_instance)
        
        labels, bbox_targets, angle_targets, _, _ = bbox_head.get_targets(
            all_level_points, batch_gt_instances, bbox_head.strides)
        
        
        flatten_labels = torch.cat(labels) # torch.Size([21824])
        bg_class_ind = len(CLASSES) # 15 表示背景
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1) 
        #------------------------------Assigner Assistant Start---------------------------------------
        with torch.no_grad():
            # 获取图像的类别提示
            filename = img_metas['img_metas'][0]['filename'].split('/')[-1]
            exist_classes_tensor = self.class_name_to_index[filename].to(t_cls_scores.device)

            # 确定选取的 pixel 数量
            count_num = int(t_cls_scores.size(0) * ratio)  # 总共需要选取的正样本数量
            # count_num = int(t_cls_scores.size(0))  # 总共需要选取的正样本数量
            teacher_probs = t_cls_scores.sigmoid()  # torch.Size([21824, 15])
            teacher_centernesses = t_centernesses.sigmoid()  # torch.Size([21824, 1])
            joint_confidence = teacher_probs * teacher_centernesses  # torch.Size([21824, 15])

            # 按联合置信度找到最大值及其类别
            max_vals, max_inds = torch.max(joint_confidence, dim=1)  # (21824,), (21824,)

            # ------------------------------------
            # 1. 输入 Prompt 类别最优先选取
            prompt_mask = torch.isin(max_inds, exist_classes_tensor)
            prompt_confidence_mask = max_vals > 0.02
            prompt_mask = prompt_mask & prompt_confidence_mask
            prompt_vals = max_vals[prompt_mask]
            prompt_inds = torch.where(prompt_mask)[0]

            # 选取 Prompt 类别中 TopK
            prompt_count = int(count_num * 0.03)
            if len(prompt_inds) > prompt_count:
                sorted_vals, sorted_inds = torch.topk(prompt_vals, prompt_count)
                prompt_inds = prompt_inds[sorted_inds]  # 最终选取的 Prompt 索引
            else:
                prompt_count = len(prompt_inds)  # 实际选取数量

            # ------------------------------------
            # 2. 整体 TopK
            overall_count = int(count_num * 0.03) - prompt_count  # 剩余需要选取的数量
            if overall_count > 0:
                non_prompt_mask = ~prompt_mask  # 非 Prompt 类别的掩码
                non_prompt_vals = max_vals[non_prompt_mask]
                non_prompt_inds = torch.where(non_prompt_mask)[0]

                sorted_vals, sorted_inds = torch.topk(non_prompt_vals, overall_count)
                overall_inds = non_prompt_inds[sorted_inds]  # 整体 TopK 的索引
            else:
                overall_inds = torch.tensor([], device=t_cls_scores.device, dtype=torch.long)

            # ------------------------------------
            # 3. 类别 TopK（Prompt 以外）
            non_prompt_classes = torch.arange(t_cls_scores.size(1), device=t_cls_scores.device)
            non_prompt_classes = non_prompt_classes[~torch.isin(non_prompt_classes, exist_classes_tensor)]  # 非 Prompt 类别

            class_topk_count = int(count_num * 0.001)
            class_topk_inds = []

            for cls in non_prompt_classes:
                cls_mask = (max_inds == cls)
                cls_vals = max_vals[cls_mask]
                cls_inds = torch.where(cls_mask)[0]

                if len(cls_inds) > class_topk_count:
                    sorted_vals, sorted_inds = torch.topk(cls_vals, class_topk_count)
                    cls_inds = cls_inds[sorted_inds]  # 选取 TopK
                class_topk_inds.append(cls_inds)

            class_topk_inds = torch.cat(class_topk_inds) if class_topk_inds else torch.tensor([], device=t_cls_scores.device)

            # 去重
            selected_inds = torch.cat([pos_inds, prompt_inds, overall_inds, class_topk_inds])
            selected_inds = torch.unique(selected_inds)  # 去除重复

            # # 合并 prompt_inds, overall_inds, class_topk_inds 并去重
            # selected_inds = torch.unique(torch.cat([prompt_inds, overall_inds, class_topk_inds]))

            # # 使用 torch.isin 过滤掉 pos_inds 中的点
            # selected_inds = selected_inds[~torch.isin(selected_inds, pos_inds)]


            # ------------------------------------
            # 初始化掩码并标记正样本
            mask = torch.zeros_like(max_vals)
            mask[selected_inds] = 1.0  # 标记为正样本

            # 计算正样本置信度和
            fg_num = mask * max_vals  # 置信度和
            fg_num = fg_num.sum()  # fg_num 是置信度和
            fg_num = max(fg_num, torch.tensor(1e-6, device=fg_num.device))
            b_mask = mask > 0.  # 转为布尔类型

        #------------------------------Assigner Assistant End-----------------------------------------

        loss_cls = QFLv2(selected_inds,
            s_cls_scores.sigmoid(),
            t_cls_scores.sigmoid(),
            weight=mask,
            reduction="sum",
        ) / fg_num
        if self.bbox_loss_type == 'l1':
            loss_bbox = (self.bbox_loss(
                s_bbox_preds[b_mask],
                t_bbox_preds[b_mask],
            ) * t_centernesses.sigmoid()[b_mask]).mean()
        else:
            all_level_points = self.prior_generator.grid_priors(
                [featmap.size()[-2:] for featmap in teacher_logits[0]],
                dtype=s_bbox_preds.dtype,
                device=s_bbox_preds.device)
            flatten_points = torch.cat(
                [points.repeat(len(teacher_logits[0][0]), 1) for points in all_level_points])
            s_bbox_preds = self.bbox_coder.decode(flatten_points, s_bbox_preds)[b_mask]
            t_bbox_preds = self.bbox_coder.decode(flatten_points, t_bbox_preds)[b_mask]
            loss_bbox = self.bbox_loss(
                s_bbox_preds,
                t_bbox_preds,
            ) * t_centernesses.sigmoid()[b_mask]
            nan_indexes = ~torch.isnan(loss_bbox)
            if nan_indexes.sum() == 0:
                loss_bbox = torch.zeros(1, device=s_cls_scores.device).sum()
            else:
                loss_bbox = loss_bbox[nan_indexes].mean()

        loss_centerness = F.binary_cross_entropy(
            s_centernesses[b_mask].sigmoid(),
            t_centernesses[b_mask].sigmoid(),
            reduction='mean'
        )

        unsup_losses = dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness
        )

        return unsup_losses


def QFLv2(selected_inds,pred_sigmoid,
          teacher_sigmoid,
          weight=None,
          beta=2.0,
          reduction='mean'):
    # all goes to 0
    pt = pred_sigmoid # (21824, 15) 表示预测的概率
    zerolabel = pt.new_zeros(pt.shape)  # (21824, 15) 所有类别都为负样本
    loss = F.binary_cross_entropy(
        pred_sigmoid, zerolabel, reduction='none') * pt.pow(beta) 
    pos = weight > 0
    print("重要依据label", len(selected_inds),'--',pos.sum())

    with open('/mnt/nas-new/home/zhanggefan/zw/mcl/result_rph/debug_txt/label.txt', "w") as file:
        # 写入 batch_inputs_all[0][0] 的形状
        file.write(f"pos : {pos.sum().cpu().numpy()}\n")
        # 写入 batch_inputs_all[0][0] 的内容
        file.write(f"pos : {pos.cpu().numpy()}\n")

    # positive goes to bbox quality 覆盖正样本的权值
    pt = teacher_sigmoid[pos] - pred_sigmoid[pos]
    loss[pos] = F.binary_cross_entropy(
        pred_sigmoid[pos], teacher_sigmoid[pos], reduction='none') * pt.pow(beta)

    valid = weight >= 0
    if reduction == "mean":
        loss = loss[valid].mean()
    elif reduction == "sum":
        loss = loss[valid].sum()
    return loss