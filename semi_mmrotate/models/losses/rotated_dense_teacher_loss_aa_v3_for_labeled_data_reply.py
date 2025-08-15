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
from mmdet.core import multi_apply
import sklearn.mixture as skm

INF = 1e8
CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
            'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
            'basketball-court', 'storage-tank', 'soccer-ball-field',
            'roundabout', 'harbor', 'swimming-pool', 'helicopter')


@ROTATED_LOSSES.register_module()
class RotatedDTLossAssignerAssistentV3forLabeledDataReply(nn.Module):
    def __init__(self, cls_channels=len(CLASSES), loss_type='origin', bbox_loss_type='l1', image_class_prompt_path='/workspace/animax/MCL/tools/Assinger_Assistent/image_class_prompt_from_chat_with_percent5_label_modified.pt'):
        super(RotatedDTLossAssignerAssistentV3forLabeledDataReply, self).__init__()
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
        
        self.cls_pool = {}


    def convert_shape(self, logits, bbox_head):
        # 先角度解码
        for i in range(len(logits[2])):
            decoder = bbox_head.angle_coder

            angle_pred = logits[2][i]
            decoded = []
            for j in range(len(angle_pred)):
                angle_pred_single = angle_pred[j]
                decoded_single = decoder.decode(angle_pred_single.permute(1, 2, 0).reshape(-1, decoder.encode_size))
                decoded_single_result = decoded_single.reshape(1, angle_pred.shape[2], angle_pred.shape[3])
                decoded.append(decoded_single_result)
            logits[2][i] = torch.stack(decoded,dim = 0)
            
        # 再合并
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


    def pre_processing(self, logits, angle_coder, alone_angle=True):
        if alone_angle:
            cls_scores, bbox_preds, angle_preds, centernesses = logits
            assert len(cls_scores) == len(bbox_preds) == len(angle_preds) == len(centernesses)
        else:
            cls_scores, bbox_preds, centernesses = logits
            assert len(cls_scores) == len(bbox_preds) == len(centernesses)

        batch_size = cls_scores[0].shape[0]   

        decode_angle_preds = []
        for angle_pred in angle_preds:
           single_lvl = []
           for i in range(batch_size):
               flatten_angle_pred = angle_pred[i].permute(1, 2, 0).reshape(-1, angle_coder.encode_size)
               de_angle_pred = angle_coder.decode(
                     flatten_angle_pred, keepdim=True).detach()
               single_lvl.append(de_angle_pred)
           single_lvl = torch.stack(single_lvl,dim = 0)
           decode_angle_preds.append(single_lvl)


        img_logits_list = []
        for img_id in range(batch_size):
            img_logits = []
            img_logits.append(torch.cat([
                x[img_id].permute(1, 2, 0).reshape(-1, self.cls_channels) for x in cls_scores
            ], dim=0))

            if alone_angle:
                img_logits.append(torch.cat([
                    torch.cat([x[img_id].permute(1, 2, 0).reshape(-1, 4), y[img_id]], dim=-1) for x, y in
                    zip(bbox_preds, decode_angle_preds)
                ], dim=0))
            else:
                img_logits.append(torch.cat([
                    x[img_id].permute(1, 2, 0).reshape(-1, 5) for x in bbox_preds
                ], dim=0))

            img_logits.append(torch.cat([
                x[img_id].permute(1, 2, 0).reshape(-1, 1) for x in centernesses
            ], dim=0))

            img_logits_list.append(img_logits)

        return img_logits_list


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
                gmm_scores[gmm_assignment == 0] = -np.inf
                indx = np.argmax(gmm_scores, axis=0)
                pos_indx = (gmm_assignment == 1) & (
                    scores >= scores[indx]).squeeze()
                pos_thr = float(scores[pos_indx].min())
                # pos_thr = max(given_gt_thr, pos_thr)
            else:
                pos_thr = given_gt_thr
        elif policy == 'middle':
            if (gmm_assignment == 1).any():
                pos_thr = float(scores[gmm_assignment == 1].min())
                # pos_thr = max(given_gt_thr, pos_thr)
            else:
                pos_thr = given_gt_thr

        return pos_thr


    def get_maxpro_classes_tensor(self, t_cls_scores, t_centernesses, s_cls_scores, s_centernesses): 
        teacher_joint_confidences = t_cls_scores * t_centernesses
        student_joint_confidences = s_cls_scores * s_centernesses
        
        teacher_joint_confidences = F.softmax(teacher_joint_confidences, dim=1)
        student_joint_confidences = F.softmax(student_joint_confidences, dim=1)
        
        teacher_max_vals, teacher_max_inds = torch.max(teacher_joint_confidences, dim=1)
        student_max_vals, student_max_inds = torch.max(student_joint_confidences, dim=1)
        
        mask_inds = (teacher_max_inds == student_max_inds)   # 师生预测类别相同
        mask_vals = (teacher_max_vals>0.7) | (student_max_vals>0.7)   # 师生预测的概率同时满足
        # 在21824每一行的15个预测上, 师生预测类别相同并且概率都大于阈值.
        mask_final = mask_inds & mask_vals   
        
        final_cls = teacher_max_inds[mask_final]
        final_cls = torch.unique(final_cls)
        
        return final_cls
                 

    # t_logits_list：[ten1[21824, 15], ten2[21824, 5], ten3[21824, 1]]
    def loss_single(self, t_logits_list, s_logits_list, batch_gt_instances_from_teacher, all_level_points, bbox_head, ratio, img_metas, level_inds):
        t_cls_scores, t_bbox_preds, t_centernesses = tuple(t_logits_list)
        s_cls_scores, s_bbox_preds, s_centernesses = tuple(s_logits_list)
        
        
        # NOTE 对于第二部分,对先验知识选取inds
        labels, _, _, _, _ = bbox_head.get_targets( all_level_points, [batch_gt_instances_from_teacher], bbox_head.strides)
        flatten_labels = torch.cat(labels) # torch.Size([21824])
        bg_class_ind = len(CLASSES) # 15 表示背景
        pos_inds = ((flatten_labels >= 0)   # 第一个inds从预测结果来的
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)    
        
        with torch.no_grad():
            teacher_probs = t_cls_scores.sigmoid()  # torch.Size([21824, 15])
            teacher_bboxes = t_bbox_preds
            teacher_centernesses = t_centernesses.sigmoid()  # torch.Size([21824, 1])
            
            
            # NOTE 开始第一部分选inds
            # P3
            teacher_probs_p3 = teacher_probs[:level_inds[0]]
            teacher_centernesses_p3 = teacher_centernesses[:level_inds[0]]
            joint_confidences_p3 = teacher_probs_p3 * teacher_centernesses_p3
            max_vals_p3 = torch.max(joint_confidences_p3, 1)[0]  # torch.Size([16384]) 保存的是每行最大值
            selected_inds_p3 = torch.topk(max_vals_p3, joint_confidences_p3.size(0))[1][:2000]   # 前2000个索引

            # P4
            teacher_probs_p4 = teacher_probs[level_inds[0]:level_inds[1]]
            teacher_centernesses_p4 = teacher_centernesses[level_inds[0]:level_inds[1]]
            joint_confidences_p4 = teacher_probs_p4 * teacher_centernesses_p4
            select_inds_p4 = torch.arange(level_inds[0], level_inds[1]).to(joint_confidences_p4.device)
            max_vals_p4 = torch.max(joint_confidences_p4, 1)[0]
            selected_inds_p4 = select_inds_p4[torch.topk(max_vals_p4, joint_confidences_p4.size(0))[1][:2000]]    # 前2000个索引

            # P5, P6, P7
            confidences_rest = teacher_probs[level_inds[1]:]
            selected_inds_rest = torch.arange(level_inds[1], teacher_probs.shape[0]).to(confidences_rest.device)

            # coarse_inds。  selected_inds_coarse：torch.Size([5344])
            selected_inds_coarse = torch.cat([selected_inds_p3, selected_inds_p4, selected_inds_rest], 0)
            
            # fine_inds由GMM来的。  all_confidences：torch.Size([21824, 15])
            all_confidences = torch.cat([joint_confidences_p3, joint_confidences_p4, confidences_rest], 0)
            max_vals = torch.max(all_confidences, 1)[0]   # max_vals.shape torch.Size([21824])   # 每行最大值
            self.mean_score = max_vals.mean()
            thres = self.gmm_policy(max_vals)
            self.thres = thres
            selected_inds_fine = torch.nonzero(max_vals > thres).squeeze(-1)

            selected_inds_one, counts = torch.cat([selected_inds_coarse, selected_inds_fine], 0).unique(return_counts=True)
            selected_inds_one = selected_inds_one[counts>1]   # 这是在coarse和fine两种情况都出现的inds
            

            # NOTE 第二部分选inds
            filename = img_metas['img_metas'][0]['filename'].split('/')[-1]
            exist_classes_tensor = self.class_name_to_index[filename].to(t_cls_scores.device)
            
            # 针对label img的lebel进行扩充处理
            if filename  not in self.cls_pool.keys(): 
                self.cls_pool[filename] = torch.zeros(15).to(t_cls_scores.device)
                self.cls_pool[filename][exist_classes_tensor] = 1
            else:
                maxpro_classes_tensor = self.get_maxpro_classes_tensor(t_cls_scores, t_centernesses, s_cls_scores, s_centernesses).to(t_cls_scores.device)
                self.cls_pool[filename][maxpro_classes_tensor] += 1
                cls_mask = (self.cls_pool[filename] - 2) > 0
                cls_indices = torch.nonzero(cls_mask, as_tuple=False).squeeze()
                
                # 确保 cls_indices 是一维张量
                if cls_indices.dim() == 0:  # 如果 cls_indices 是标量张量
                    cls_indices = cls_indices.unsqueeze(0)  # 转换为一维张量
                                
                if torch.numel(cls_indices) != 0:
                    # print(type(exist_classes_tensor))
                    # print(exist_classes_tensor.shape)
                    # print(exist_classes_tensor)
                    # print(type(cls_indices))
                    # print(cls_indices)
                    # print(cls_indices.shape)
                    # print('done')
                    exist_classes_tensor = torch.cat((exist_classes_tensor, cls_indices), dim=0)
                    
                
            
            joint_confidence = teacher_probs * teacher_centernesses  # torch.Size([21824, 15])
            # 按联合置信度找到每一行的最大值及其类别
            max_vals, max_inds = torch.max(joint_confidence, dim=1)  # (21824,), (21824,)
            # 确定选取的 pixel 数量
            count_num = int(t_cls_scores.size(0) * 0.5)  # TODO 这里取代了ratio参数,控制总共需要选取的正样本数量

            # ------------------------------------
            # 1. 输入 Prompt 类别最优先选取
            prompt_mask = torch.isin(max_inds, exist_classes_tensor)
            # prompt_confidence_mask = max_vals > 0.02   # 这里的阈值可以使用EM的阈值
            prompt_confidence_mask = max_vals > thres
            # prompt_mask = prompt_mask & prompt_confidence_mask
            prompt_mask = prompt_mask | prompt_confidence_mask
            prompt_vals = max_vals[prompt_mask]
            prompt_inds = torch.where(prompt_mask)[0]
            # 选取 Prompt 类别中 TopK
            prompt_count = int(count_num * 0.2)
            if len(prompt_inds) > prompt_count:
                sorted_vals, sorted_inds = torch.topk(prompt_vals, prompt_count)
                prompt_inds = prompt_inds[sorted_inds]  # 最终选取的 Prompt 索引
            else:
                prompt_count = len(prompt_inds)  # 实际选取数量

            # ------------------------------------
            # 2. 整体 TopK   NOTE 这里需要核查, prompt_inds为0
            # overall_count = int(count_num * 0.03) - prompt_count  # 剩余需要选取的数量
            # if overall_count > 0:
            #     non_prompt_mask = ~prompt_mask  # 非 Prompt 类别的掩码
            #     non_prompt_vals = max_vals[non_prompt_mask]
            #     non_prompt_inds = torch.where(non_prompt_mask)[0]

            #     sorted_vals, sorted_inds = torch.topk(non_prompt_vals, overall_count)
            #     overall_inds = non_prompt_inds[sorted_inds]  # 整体 TopK 的索引
            # else:
            #     overall_inds = torch.tensor([], device=t_cls_scores.device, dtype=torch.long)
            # ------------------------------------
            # 2. 整体 TopK   NOTE 这里需要核查, prompt_inds为0
            overall_count = int(count_num * 0.2)
            non_prompt_mask = ~prompt_mask  # 非 Prompt 类别的掩码
            non_prompt_vals = max_vals[non_prompt_mask]
            non_prompt_inds = torch.where(non_prompt_mask)[0]
            if len(non_prompt_inds) > overall_count:
                sorted_vals, sorted_inds = torch.topk(non_prompt_vals, overall_count)
                overall_inds = non_prompt_inds[sorted_inds]  # 整体 TopK 的索引
            else:
                overall_inds = non_prompt_inds

            # ------------------------------------
            # 3. 各个类别 TopK（Prompt 以外）
            non_prompt_classes = torch.arange(t_cls_scores.size(1), device=t_cls_scores.device)
            # non_prompt_classes = non_prompt_classes[~torch.isin(non_prompt_classes, exist_classes_tensor)]  # 非 Prompt 类别
            class_topk_count = int(count_num * 0.04)
            class_topk_inds = []   # NOTE 注意这里class_topk_inds是空
            for cls in non_prompt_classes:
                cls_mask = (max_inds == cls)
                cls_vals = max_vals[cls_mask]
                cls_inds = torch.where(cls_mask)[0]   # 选取为true的位置

                if len(cls_inds) > class_topk_count:
                    sorted_vals, sorted_inds = torch.topk(cls_vals, class_topk_count)
                    cls_inds = cls_inds[sorted_inds]  # 选取 TopK
                class_topk_inds.append(cls_inds)
            class_topk_inds = torch.cat(class_topk_inds) if class_topk_inds else torch.tensor([], device=t_cls_scores.device)

            # 去重  
            selected_inds_two = torch.cat([pos_inds, prompt_inds, overall_inds, class_topk_inds])
            selected_inds_two = torch.unique(selected_inds_two)  # 去除重复 29
            selected_inds_two = selected_inds_two.to(torch.long)            
            
            
            selected_inds, counts = torch.cat([selected_inds_one, selected_inds_two], 0).unique(return_counts=True)
            selected_inds = selected_inds[counts>1]   # 这是在coarse和fine两种情况都出现的inds            
            weight_mask = torch.zeros_like(max_vals)
            weight_mask[selected_inds] = max_vals[selected_inds]
            b_mask = weight_mask > 0.   # 在coarse和fine里面都出现过的inds. 而且要求预测结果大于0
            
            
        if b_mask.sum() == 0:
            loss_cls = QFLv2(   # 3个变量的维度相同
                s_cls_scores.sigmoid(),
                teacher_probs,
                weight=max_vals,
                reduction="sum",
            ) / max_vals.sum()
            loss_bbox = torch.zeros(1).squeeze().to(max_vals.device)
            loss_centerness = torch.zeros(1).squeeze().to(max_vals.device)
        else:
            loss_cls = QFLv2(
                s_cls_scores.sigmoid(),
                teacher_probs,
                weight=weight_mask,
                reduction="sum",
            ) / weight_mask.sum()

            loss_bbox = (self.bbox_loss(
                s_bbox_preds[b_mask],
                teacher_bboxes[b_mask],
            ) * weight_mask[:, None][b_mask]).mean() * 10

            loss_centerness = (F.binary_cross_entropy(
                s_centernesses[b_mask].sigmoid(),
                teacher_centernesses[b_mask],
                reduction='none'
            )* weight_mask[:, None][b_mask]).mean() * 10

        return loss_cls, loss_bbox, loss_centerness


    # 我要融合GMM和RSST两种选inds的方法
    def forward(self, teacher_logits, student_logits, ratio=0.03, img_metas=None, bbox_head=None, **kwargs):
        # 在pre_processing中处理模型生成结果.
        img_t_logits_list = self.pre_processing(teacher_logits[:4], bbox_head.angle_coder)
        img_s_logits_list = self.pre_processing(student_logits[:4], bbox_head.angle_coder)
        # T:list[list1, list2]。list1:list[tensor1[21824, 15], tensor2[21824, 5], tensor3[21824, 1]]。list2:list[tensor1[21824, 15], tensor2[21824, 5], tensor3[21824, 1]]
        featmap_sizes = [featmap.size()[-2:] for featmap in teacher_logits[0]]
        level_inds = []
        start = 0
        for size in featmap_sizes:
            start = start + size[0] * size[0]
            level_inds.append(start)
        level_inds = level_inds[:2]   # [16384, 20480]
        
        all_level_points = self.prior_generator.grid_priors([featmap.size()[-2:] for featmap in teacher_logits[0]])

        # get labels and bbox_targets of each image 每张图片单独处理,并且处理的inds不一样. 对于每张图片,从教师的预测中选取伪标签
        losses_list = multi_apply(
                            self.loss_single,
                            img_t_logits_list,
                            img_s_logits_list,
                            teacher_logits[4],
                            all_level_points=all_level_points,
                            bbox_head = bbox_head,
                            ratio = ratio,
                            img_metas = img_metas,
                            level_inds=level_inds)

        unsup_losses = dict(
            loss_cls=sum(losses_list[0]) / len(losses_list[0]),   # 除以图片的数量,计算平均loss
            loss_bbox=sum(losses_list[1]) / len(losses_list[1]),
            loss_centerness=sum(losses_list[2]) / len(losses_list[2])
        )

        return unsup_losses        

def QFLv2(pred_sigmoid,
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