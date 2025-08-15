# Copyright (c) OpenMMLab. All rights reserved.
import os, copy, math

import torch
import torch.nn as nn
from mmcv.cnn import Scale, ConvModule
from mmcv.runner import force_fp32
from mmrotate.models.dense_heads.rotated_anchor_free_head import RotatedAnchorFreeHead
from mmdet.core import multi_apply, reduce_mean

from mmrotate.models.builder import ROTATED_HEADS, build_loss
from mmrotate.core import build_bbox_coder, multiclass_nms_rotated
from mmrotate.models.losses.gaussian_dist_loss import xy_wh_r_2_xy_sigma, gwd_loss
from mmdet.core.anchor.point_generator import MlvlPointGenerator
import sklearn.mixture as skm
import numpy as np

INF = 1e8


@ROTATED_HEADS.register_module()
class SemiMixHead1SparseAll(RotatedAnchorFreeHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    Compared with FCOS head, Rotated FCOS head add a angle branch to
    support rotated object detection.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        angle_version (str): Angle representations. Defaults to 'le90'.
        use_hbbox_loss (bool): If true, use horizontal bbox loss and
            loss_angle should not be None. Default to False.
        scale_angle (bool): If true, add scale to angle pred branch.
            Default to True.
        angle_coder (:obj:`ConfigDict` or dict): Config of angle coder.
        h_bbox_coder (dict): Config of horzional bbox coder,
            only used when use_hbbox_loss is True.
        bbox_coder (:obj:`ConfigDict` or dict): Config of bbox coder. Defaults
            to 'DistanceAnglePointCoder'.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_angle (:obj:`ConfigDict` or dict, Optional): Config of angle loss.

    Example:
        >>> self = RotatedFCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, angle_pred = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 strides=[8, 16, 32, 64, 128],
                 strides_single = [8],
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                      (512, INF)),
                 centerness_on_reg=True,
                 center_sampling=True,
                 center_sample_radius=1.5,
                 center_sample_radius_p=0.75,
                 norm_on_bbox=False,
                 angle_version='le90',
                 edge_loss_start_iter=60000,
                 joint_angle_start_iter=10000,
                 voronoi_type='gaussian-orientation',
                 voronoi_thres=dict(
                     default=[0.994, 0.005],
                     override=(([2, 11], [0.999, 0.6]),
                               ([7, 8, 10, 14], [0.95, 0.005]))),
                 square_cls=[1, 9, 11],
                 edge_loss_cls=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13],
                 post_process={11: 1.2},
                 bbox_coder=dict(type='DistanceAnglePointCoder'),
                 angle_coder=dict(
                     type='PSCCoder',
                     angle_version='le90',
                     dual_freq=False,
                     num_step=3,
                     thr_mod=0),
                 loss_cls=dict(
                     type='mmdet.FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='RotatedIoULoss', loss_weight=1.0),
                 loss_cent=dict(
                     type='mmdet.L1Loss', loss_weight=0.01),
                 loss_hbox=dict(
                     type='mmdet.IoULoss', loss_weight=1.0),
                 loss_overlap=dict(
                     type='GaussianOverlapLoss', loss_weight=100.0),
                 loss_voronoi=dict(
                     type='GaussianVoronoiLoss', loss_weight=50.0),
                 loss_bbox_edg=dict(
                     type='EdgeLoss', loss_weight=10.0),
                 loss_ss=dict(
                     type='mmdet.SmoothL1Loss', loss_weight=0.2, beta=0.1),
                 gwd_weight = 1.0, 
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=[
                         dict(
                             type='Normal',
                             name='conv_cls',
                             std=0.01,
                             bias_prob=0.01)
                         #dict(
                         #    type='Normal',
                         #    name='conv_gate',
                         #    std=0.01,
                         #    bias_prob=0.01)
                                  ]),
                 **kwargs):
        self.angle_coder = build_bbox_coder(angle_coder)
        super().__init__(
            num_classes,
            in_channels,
            strides=strides,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        #self.strides_full = strides_full
        self.centerness_on_reg = centerness_on_reg
        self.norm_on_bbox = norm_on_bbox
        self.regress_ranges = regress_ranges
        #self.regress_ranges_full = regress_ranges_full
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.center_sample_radius_p = center_sample_radius_p
        self.angle_version = angle_version
        self.edge_loss_start_iter = edge_loss_start_iter
        self.joint_angle_start_iter = joint_angle_start_iter
        self.voronoi_thres = voronoi_thres
        self.voronoi_type = voronoi_type
        self.square_cls = square_cls
        self.edge_loss_cls = edge_loss_cls
        self.post_process = post_process
        self.loss_ss = build_loss(loss_ss)
        self.gwd_weight = gwd_weight
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_hbox = build_loss(loss_hbox)
        self.loss_cent = build_loss(loss_cent)
        self.loss_overlap = build_loss(loss_overlap)
        self.loss_voronoi = build_loss(loss_voronoi)
        self.loss_bbox_edg = build_loss(loss_bbox_edg)
        self.prior_generator_single = MlvlPointGenerator(strides_single)

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in [8, 16, 32, 64, 128]])
        self.conv_angle = nn.Conv2d(
            self.feat_channels, self.angle_coder.encode_size, 3, padding=1)
        #self.conv_gate = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        ######init_cfg!!!!!!!!!!!!!!!!!!!!!!
        # self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        # if self.is_scale_angle:
        #    self.scale_angle = Scale(1.0)
        
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

    def nested_projection(self, pred, target):
        target_xy1 = target[..., 0:2] - target[..., 2:4] / 2
        target_xy2 = target[..., 0:2] + target[..., 2:4] / 2
        target_projected = torch.cat((target_xy1, target_xy2), -1)
        pred_xy = pred[..., 0:2]
        pred_wh = pred[..., 2:4]
        da = (pred[..., 4] - target[..., 4]).detach()
        cosa = torch.cos(da).abs()
        sina = torch.sin(da).abs()
        pred_wh = torch.matmul(
            torch.stack((cosa, sina, sina, cosa), -1).view(*cosa.shape, 2, 2),
            pred_wh[..., None])[..., 0]
        pred_xy1 = pred_xy - pred_wh / 2
        pred_xy2 = pred_xy + pred_wh / 2
        pred_projected = torch.cat((pred_xy1, pred_xy2), -1)
        return pred_projected, target_projected

    def forward(self, x, get_data=True):
        """Forward features from the upstream network.
        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                angle_preds (list[Tensor]): Box angle for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
                centernesses (list[Tensor]): centerness for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
        """

       
        cls_score, bbox_pred, angle_pred, centerness = multi_apply(self.forward_single,
                                                                   x,
                                                                   [get_data,get_data,get_data,get_data,get_data],
                                                                   self.scales,
                                                                   self.strides)
        
        return cls_score, bbox_pred, angle_pred, centerness

       

    def forward_single(self, x, get_data, scale, stride):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.
        Returns:
            tuple: scores for each class, bbox predictions, angle predictions \
                and centerness predictions of input feature maps.
        """
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.conv_cls(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)

        
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        
        if self.norm_on_bbox:
            # bbox_pred needed for gradient computation has been modified
            # by F.relu(bbox_pred) when run with PyTorch 1.10. So replace
            # F.relu(bbox_pred) with bbox_pred.clamp(min=0)
            #bbox_pred = bbox_pred.clamp(min=0)
            bbox_pred = bbox_pred.exp()
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()

        angle_pred = self.conv_angle(reg_feat)
        # if get_data:
        #     angle_pred = \
        #     self.angle_coder.decode(angle_pred.reshape(-1, self.angle_coder.encode_size)).reshape(angle_pred.shape[0], 1, angle_pred.shape[2], angle_pred.shape[3])
        # if self.is_scale_angle:
        #    angle_pred = self.scale_angle(angle_pred).float()
        return cls_score, bbox_pred, angle_pred, centerness


    def select_topk_gmm(self, start_idx, end_idx, use_centerness, joint_confidence_all, mask, topk_num=2000):
        
        with torch.no_grad():
            
            topk_num = int(topk_num)
            # layer_probs = teacher_probs[start_idx:end_idx]
            # if use_centerness:
            #     layer_centernesses = teacher_centernesses[start_idx:end_idx]
            #     layer_confidences = layer_probs * layer_centernesses
            # else:
            #     layer_confidences = layer_probs
            
            layer_confidences = joint_confidence_all[start_idx:end_idx]

            max_vals, _ = torch.max(layer_confidences, 1)
            layer_mask = mask[start_idx:end_idx]
            
            layer_vals = max_vals[layer_mask]
            layer_inds = torch.where(layer_mask)[0] + start_idx

            # selected_indices_coarse = torch.empty(0, dtype=torch.long, device=joint_confidence_all.device)
        
            # topk_num = min(topk_num, layer_vals.size(0))
            # if topk_num > 0:
            #     _, topk_indices = torch.topk(layer_vals, topk_num)
            #     selected_indices_coarse = layer_inds[topk_indices]
                    
            # self.mean_score.append(layer_vals.mean())
            thres = self.gmm_policy(layer_vals)
            # self.thres.append(thres)

            fine_mask = layer_vals < thres
            selected_inds_fine = layer_inds[fine_mask]
            # selected_indices, counts = torch.cat([selected_indices_coarse, selected_inds_fine], dim=0).unique(return_counts=True)
            # selected_indices = selected_indices.long()
            # selected_indices = selected_indices[counts > 1]
            
        return selected_inds_fine


    def get_gmm_indices(self, tensor_single_img, level_inds):
        
        with torch.no_grad():

            joint_confidence_all = tensor_single_img
            max_vals, max_inds = torch.max(joint_confidence_all, 1)
            
            layer_configs = [
                (0, level_inds[0], True, level_inds[0] * 0.12),  # P3
                (level_inds[0], level_inds[0]+level_inds[1], True, (level_inds[1]) * 0.24),  # P4
                (level_inds[0]+level_inds[1], tensor_single_img.size(0), False, (tensor_single_img.size(0) - level_inds[0] - level_inds[1]))  # P5+
            ]
            '''
            selected_prompt_indices = torch.empty(0, device=teacher_probs.device)
            for start_idx, end_idx, use_centerness, topk_num in layer_configs:
                selected_prompt_indices_layers = self.select_topk_gmm(start_idx, end_idx, use_centerness, teacher_probs, 
                                                          teacher_centernesses, prompt_mask, topk_num=topk_num)
                
                selected_prompt_vals_layers = max_vals[selected_prompt_indices_layers]
                thres_prompt = self.gmm_policy(selected_prompt_vals_layers, policy=self.policy)
                mask = selected_prompt_vals_layers > thres_prompt
                selected_prompt_indices_layers = selected_prompt_indices_layers[mask]
    
                selected_prompt_indices = torch.cat([selected_prompt_indices, selected_prompt_indices_layers], dim=0).long()
                
            '''
            mask_ = torch.ones_like(max_inds, dtype=torch.bool)
            selected_indices = torch.empty(0, device=tensor_single_img.device)
            for start_idx, end_idx, use_centerness, topk_num in layer_configs:
                selected_indices_layers = self.select_topk_gmm(start_idx, end_idx, use_centerness, joint_confidence_all, mask_, topk_num=topk_num)
                
                selected_vals_layers = max_vals[selected_indices_layers]
                thres = self.gmm_policy(selected_vals_layers)
                mask = selected_vals_layers < thres  # 这里取小的
                selected_indices_layers = selected_indices_layers[mask]
                
                selected_indices = torch.cat([selected_indices, selected_indices_layers], dim=0).long()
        
        return selected_indices
    
    
    def get_gmm_indices1(self, tensor_single_img, level_inds):  
        '''
            不分层，直接获取整个tensor的GMM结果
        '''    
        with torch.no_grad():

            joint_confidence_all = tensor_single_img
            max_vals, max_inds = torch.max(joint_confidence_all, 1)
            
            thres = self.gmm_policy(max_vals)
            mask = max_vals < thres  # 这里取小的
            selected_indices_layers = max_inds[mask]
    
        return selected_indices_layers



    def get_neg(self, pos_inds, flatten_centerness, flatten_cls_scores, featmap_sizes, num_imgs, strict=True):
        """
        将按层级批次排列的置信度张量重组为按图像排列，并对每张图像取 top-k 置信度，
        然后将这些 top-k 锚点的索引映射回原始张量。
        
        Args:
            joint_confidence_all (torch.Tensor): 原始的联合置信度张量。
            num_imgs (int): 图像总数（单批次，即原始代码中的 num_imgs）。
            featmap_sizes (list): 包含每个特征图尺寸的列表。
            k (int): 每张图像需要获取的 top-k 锚点数量。
            
        Returns:
            list: 包含每张图像 top-k 锚点原始索引的列表。
                例如: [[img0_idx0, img0_idx1, ...], [img1_idx0, img1_idx1, ...], ...]
        """
        with torch.no_grad():
            pred_centerness = flatten_centerness.sigmoid()
            pred_cls_scores = flatten_cls_scores.sigmoid()
            pred_centerness = pred_centerness.unsqueeze(1) 
            joint_confidence_all = pred_centerness * pred_cls_scores
        
            # 1. 计算每一层的锚点个数和总锚点数
            level_inds = [size[0] * size[0] for size in featmap_sizes]
            num_levels = len(level_inds)
            total_anchors_per_image = sum(level_inds)
            
            # 2. 重组张量，将数据从“按层级批次”排列转换为“按图像”排列
            new_tensor_by_image_list = []
            
            for i in range(2 * num_imgs):  # 遍历每一张图片
                batch_idx = i // num_imgs
                img_idx_in_batch = i % num_imgs
                
                conf_for_this_image = []
                
                for level_idx, level_size in enumerate(level_inds):   # 遍历每一张图片的每个level
                    # 计算当前层级在总张量中的起始位置
                    batch_start_idx = batch_idx * (num_imgs * total_anchors_per_image)
                    level_start_in_batch = sum(num_imgs * level_inds[l] for l in range(level_idx))
                    img_start_in_level = img_idx_in_batch * level_size
                    
                    start_idx = batch_start_idx + level_start_in_batch + img_start_in_level
                    end_idx = start_idx + level_size
                    if level_idx < 2:  # 在0，1的时候选取联合置信度
                        conf_for_this_image.append(joint_confidence_all[start_idx: end_idx])
                    else:  # 在2，3，4的时候选取分类置信度，因为中心度已经不重要了
                        conf_for_this_image.append(pred_cls_scores[start_idx: end_idx])
                    
                image_confidence = (torch.cat(conf_for_this_image, dim=0)).to(pos_inds.device) 
                new_tensor_by_image_list.append(image_confidence)

            new_tensor_by_image = (torch.stack(new_tensor_by_image_list, dim=0)).to(pos_inds.device) 

            # 3. 对每张图像取 top-k 并进行索引映射
            all_original_indices = []
            
            for img_idx in range(2 * num_imgs):
                # 对重组后的单张图像张量
                level_ind_single = level_inds[0:2]
                new_indices_in_image = self.get_gmm_indices1(new_tensor_by_image[img_idx], level_ind_single)
                
                original_indices_for_image = []
                for anchor_in_img_idx in new_indices_in_image:
                    # 找到 anchor_in_img_idx 所在的层级 (level)
                    level_idx = 0
                    level_start_anchor_idx = 0
                    for level_size in level_inds:
                        if anchor_in_img_idx < level_start_anchor_idx + level_size:
                            break
                        level_start_anchor_idx += level_size
                        level_idx += 1
                    
                    # 计算锚点在当前层级内的偏移量 (offset)
                    offset_in_level = anchor_in_img_idx - level_start_anchor_idx
                    
                    # 计算原始张量中的层级起始位置
                    original_level_start_idx = sum(num_imgs * level_inds[l] for l in range(level_idx))
                    
                    # 计算原始索引
                    batch_idx = img_idx // num_imgs
                    img_idx_in_batch = img_idx % num_imgs
                    original_idx = (batch_idx * num_imgs * total_anchors_per_image) + \
                                original_level_start_idx + \
                                (img_idx_in_batch * level_inds[level_idx]) + \
                                offset_in_level
                    
                    original_indices_for_image.append(original_idx)
                    
                all_original_indices.append(torch.tensor(original_indices_for_image, dtype=torch.long))
                
            merged_tensor = torch.cat(all_original_indices)
            unique_tensor = torch.unique(merged_tensor).to(pos_inds.device) 

        return unique_tensor

    # --- 示例用法 ---
    # 假设你的数据如下
    # num_imgs = 4
    # featmap_sizes = [[32, 32], [16, 16], [8, 8], [4, 4], [2, 2]]
    # level_inds = [1024, 256, 64, 16, 4]
    # total_anchors = sum(level_inds) * 2 * num_imgs
    # joint_confidence_all = torch.rand(total_anchors, 1) # 假设的随机数据

    # k = 100
    # topk_original_indices = get_topk_original_indices(joint_confidence_all, num_imgs, featmap_sizes, k)

    # print(f"每张图像的 top-{k} 锚点原始索引列表：")
    # for i, indices in enumerate(topk_original_indices):
    #     print(f"图像 {i}: {indices.shape}")
    
     


    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds'))
    def loss(
            self,
            cls_scores,
            bbox_preds,
            angle_preds,
            centernesses,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=None,
    ):
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            angle_preds (list[Tensor]): Box angle for each scale level, each \
                is a 4D-tensor, the channel number is num_points * encode_size.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        
        assert len(cls_scores) == len(bbox_preds) == len(angle_preds) == len(centernesses)
        # 4,15,(176,176),(88,88),(44,44),(22,22),(11,11)     4,4     4,3   4,1       
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        
        # bbox_targets here is in format t,b,l,r
        # angle_targets is not coded here
        point_gt_list = []
        hr_gt_list = []
        
        for data_sample in batch_gt_instances:
            point_mask = data_sample['labels'][:, 1] == 2
            point_gt = {}
            hr_gt = {}
            for key in data_sample.keys():
                point_gt[key] = data_sample[key][point_mask]
                hr_gt[key] = data_sample[key][~point_mask]
            point_gt_list.append(point_gt)
            hr_gt_list.append(hr_gt)

        

        labels, bbox_targets, angle_targets, bid_targets, centernesses_targets = self.get_targets(
            all_level_points, point_gt_list, self.strides, use_single=True)
        

        hr_labels, hr_bbox_targets, hr_angle_targets, \
            hr_bid_targets, hr_centerness_targets = self.get_targets(
            all_level_points, hr_gt_list, self.strides, use_single=False)

       
        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds, angle_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(-1, self.angle_coder.encode_size)
            for angle_pred in angle_preds
        ]
        
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        
        # 这里是预测结果
        flatten_centerness = torch.cat(flatten_centerness + flatten_centerness)         # torch.Size([330088])
        flatten_cls_scores = torch.cat(flatten_cls_scores + flatten_cls_scores)       # torch.Size([330088, 15])
        flatten_bbox_preds = torch.cat(flatten_bbox_preds + flatten_bbox_preds)       # torch.Size([330088, 4])
        flatten_angle_preds = torch.cat(flatten_angle_preds + flatten_angle_preds)    # torch.Size([330088, 3])

        # 这里是GT
        flatten_labels = torch.cat(labels + hr_labels)[:, 0]                # torch.Size([330088])                       
        flatten_ws = torch.cat(labels + hr_labels)[:, 1]                    # torch.Size([330088])
        flatten_bbox_targets = torch.cat(bbox_targets + hr_bbox_targets)    # torch.Size([330088, 4])
        flatten_angle_targets = torch.cat(angle_targets + hr_angle_targets) # torch.Size([330088, 1])
        flatten_bid_targets = torch.cat(bid_targets + hr_bid_targets)       # torch.Size([330088, 4])
        flatten_centerness_targets = torch.cat(centernesses_targets+hr_centerness_targets)  # torch.Size([330088])
        
        all_level_points = all_level_points + all_level_points   # 这是一张图片的所有先验点？？？点标注和框标注分开
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])   # torch.Size([330088, 2])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        # 除类别loss，其他loss使用这里的sparse GT标注，对于稀疏标注情景下的未标注实例没有选择。这肯定是不对的（没有充分利用信息）。应该使用sparse GT + 除GT外GMM的正
        # 但是使用GMM的正，在sup分支没有对应的GT部分。所以能不能考虑使用GMM的负，这部分在GT中肯定有对应。毕竟对于背景的学习错误也是一种知识。
        # TODO 原始的半监督没有使用背景的学习，这里先不加入背景的学习。
        pos_inds = ((flatten_labels >= 0)   # torch.Size([45])
                    # & (flatten_labels != 2)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        
        # GMM处理预测结果。基于以上信息，根据flatten_centerness和flatten_cls_scores得到GMM判为背景的索引，定义一个bool表示使用GMM分到的负，还是使用负的一半。
        # 输出：GMM算法，分图片分层得到的
        # neg_tensor = self.get_neg(pos_inds, flatten_centerness, flatten_cls_scores, featmap_sizes, num_imgs, strict=True)
        
        
        # 极简版本的筛选，先看看效果
        pred_probs = flatten_cls_scores.sigmoid()
        pred_centernesses = flatten_centerness.sigmoid()
        pred_centernesses = pred_centernesses.unsqueeze(1) 
        pred_confidence_all = pred_probs * pred_centernesses
        
        max_vals, max_inds = torch.max(pred_confidence_all, 1)
        max_vals_copy = max_vals.detach().clone()
        thres = self.gmm_policy(max_vals_copy)
        fine_mask = max_vals < thres
        neg_tensor = max_inds[fine_mask]
    
        
        loss_cls_index = torch.cat((pos_inds, neg_tensor), dim=0)

        pos_centerness_targets = flatten_centerness_targets[pos_inds] 
        pos_centerness = flatten_centerness[pos_inds]  

        
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)

        # loss_cls = self.loss_cls(
        #         flatten_cls_scores[pos_inds], flatten_labels[pos_inds], avg_factor=num_pos)
        # TODO 类别loss使用全部索引。这肯定是不对的（利用了错误的类别指导信息）。这里送进去的应该是sparse GT + 除GT外GMM的负。
        loss_cls = self.loss_cls(  
            flatten_cls_scores[loss_cls_index], flatten_labels[loss_cls_index], avg_factor=num_pos)  # flatten_cls_scores torch.Size([330088, 15])      flatten_labels.shape torch.Size([330088])

        pos_cls_scores = flatten_cls_scores[pos_inds].sigmoid()
        pos_labels = flatten_labels[pos_inds]
        pos_cls_scores = torch.gather(pos_cls_scores, 1, pos_labels[:, None])[:, 0]
        
        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_angle_preds = flatten_angle_preds[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_angle_targets = flatten_angle_targets[pos_inds]
        pos_bid_targets = flatten_bid_targets[pos_inds]
        pos_ws = flatten_ws[pos_inds]
       
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)


        self.vis = [None] * len(batch_gt_instances)  # For visual debug
        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_labels = flatten_labels[pos_inds]

            mask_point = pos_ws == 2
            
            
            
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
          

            pos_decoded_angle_preds = self.angle_coder.decode(
                pos_angle_preds, keepdim=True)
            if self.iter_count < self.joint_angle_start_iter:
                pos_decoded_angle_preds = pos_decoded_angle_preds.detach()
            square_mask = torch.zeros_like(pos_labels, dtype=torch.bool)
            for c in self.square_cls:
                square_mask = torch.logical_or(square_mask, pos_labels == c)
            pos_decoded_angle_preds[square_mask] = 0
            target_mask = torch.abs(
                pos_angle_targets[square_mask]) < torch.pi / 4
            pos_angle_targets[square_mask] = torch.where(
                target_mask, 0, -torch.pi / 2)
            
            pos_bbox_targets = torch.cat([pos_bbox_targets, pos_angle_targets],
                                         dim=-1)
            
            
            
            #pos_cent_preds = pos_points + pos_bbox_preds[:, 2:]
            pos_rbox_targets = self.bbox_coder.decode(pos_points, pos_bbox_targets)  # Key. targets[:, -1] must be zero
            #pos_rbox_preds = torch.cat((pos_rbox_targets[:, :2], pos_bbox_preds[:, :2] * 2, pos_decoded_angle_preds),
            #                           -1)
            pos_rbox_preds = self.bbox_coder.decode(pos_points,
                                                    torch.cat((pos_bbox_preds, pos_decoded_angle_preds), -1))
           
            cos_r = torch.cos(pos_decoded_angle_preds)
            sin_r = torch.sin(pos_decoded_angle_preds)
            R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1)[mask_point].reshape(-1, 2, 2)
            #pos_gaus_preds = R.matmul(torch.diag_embed(pos_bbox_preds[:, :2])).matmul(R.permute(0, 2, 1))
            
            pos_gaus_preds = R.matmul(torch.diag_embed(pos_rbox_preds[mask_point][:, 2:4] * 0.5)).matmul(R.permute(0, 2, 1))
           
            num_p = len(pos_gaus_preds)
           

            loss_cent = self.loss_cent(
                # pos_cent_preds,
                pos_rbox_preds[:, :2],
                pos_rbox_targets[:, :2],
                weight=(pos_ws == 2)[:, None] * pos_centerness_targets[:, None],
                avg_factor=num_pos)

            loss_hbox = self.loss_hbox(
                *self.nested_projection(pos_rbox_preds,
                                        pos_rbox_targets),
                weight=(pos_ws == 1) * pos_centerness_targets,
                avg_factor=centerness_denorm)

            loss_bbox = self.loss_bbox(
                pos_rbox_preds,
                pos_rbox_targets,
                weight=(pos_ws == 0),
                avg_factor=num_pos)

            
            loss_bbox_vor = pos_rbox_targets.new_tensor(0)
            loss_bbox_ovl = pos_rbox_targets.new_tensor(0)
            loss_bbox_edg = pos_rbox_targets.new_tensor(0)
            if num_p > 0:
                # Aggregate targets of the same instance based on their identical bid
                bid_with_view = pos_bid_targets[mask_point][:, 3] + \
                              0.5 * pos_bid_targets[mask_point][:, 2]
                bid, idx = torch.unique(bid_with_view, return_inverse=True)

                # Generate a mask to eliminate bboxes without correspondence
                # (bcnt is supposed to be 3, for ori, rot, and flp)
                ins_bid_with_view = bid.new_zeros(*bid.shape).index_reduce_(
                    0, idx, bid_with_view, 'amin', include_self=False)
                _, bidx, bcnt = torch.unique(
                    ins_bid_with_view.long(),
                    return_inverse=True,
                    return_counts=True)
                bmsk = bcnt[bidx] == 2
                # Select instances by batch
                ins_bids = pos_bid_targets.new_zeros(*bid.shape).index_reduce_(
                    0, idx, pos_bid_targets[mask_point][:, 3], 'amin', include_self=False)

                ins_batch = pos_bid_targets.new_zeros(*bid.shape).index_reduce_(
                    0, idx, pos_bid_targets[mask_point][:, 0], 'amin', include_self=False)

                ins_labels = pos_labels.new_zeros(*bid.shape).index_reduce_(
                    0, idx, pos_labels[mask_point], 'amin', include_self=False)
                
                #ins_ws = pos_labels.new_zeros(*bid.shape).index_reduce_(
                #    0, idx, pos_ws, 'amin', include_self=False)
                #mask_point = ins_ws == 2

                ins_gaus_preds = pos_gaus_preds.new_zeros(
                    *bid.shape, 4).index_reduce_(
                    0, idx, pos_gaus_preds.view(-1, 4), 'mean',
                    include_self=False).view(-1, 2, 2)
                

                #ins_rbox_preds = pos_rbox_preds.new_zeros(
                #    *bid.shape, pos_rbox_preds.shape[-1]).index_reduce_(
                #    0, idx, pos_rbox_preds[mask_point], 'mean',
                #    include_self=False)

                ins_rbox_targets = pos_rbox_targets.new_zeros(
                    *bid.shape, pos_rbox_targets.shape[-1]).index_reduce_(
                    0, idx, pos_rbox_targets[mask_point], 'mean',
                    include_self=False)
                
                pair_cls_scores = torch.empty(
                *bid.shape, device=bid.device).index_reduce_(
                    0, idx, pos_cls_scores[mask_point], 'mean',
                    include_self=False)[bmsk].view(-1, 2)
            
                pair_gaus_preds = ins_gaus_preds[bmsk].view(-1, 2, 2, 2)
                bbox_area = pair_gaus_preds[:, 0, 0, 0] * pair_gaus_preds[:, 0, 1, 1] * 4
                
                ss_info = batch_img_metas[0]['ss']
                valid_p = pair_cls_scores[:, 1] > 0.1
                
                bbox_area = pair_gaus_preds[:, 0, 0, 0] * pair_gaus_preds[:, 0, 1, 1] * 4
                sca = ss_info[1] if ss_info[0] == 'sca' else 1
                valid_p = torch.logical_and(valid_p, bbox_area > 24 ** 2)
                valid_p = torch.logical_and(valid_p, bbox_area * sca > 24 ** 2)
                valid_p = torch.logical_and(valid_p, bbox_area < 512 ** 2)
                valid_p = torch.logical_and(valid_p, bbox_area * sca < 512 ** 2)

                ori_mu_all = ins_rbox_targets[:, 0:2]
                
               
                
                for batch_id in range(len(batch_gt_instances)):
                    group_mask = (ins_batch == batch_id) & (ins_bids != 0)
                    # Overlap and Voronoi Losses
                    mu = ori_mu_all[group_mask]
                    sigma = ins_gaus_preds[group_mask]
                    label = ins_labels[group_mask]#######
                    if len(mu) >= 2:
                        loss_bbox_ovl += self.loss_overlap((mu, sigma.bmm(sigma)))
                    if len(mu) >= 1:
                        pos_thres = [self.voronoi_thres['default'][0]] * self.num_classes
                        neg_thres = [self.voronoi_thres['default'][1]] * self.num_classes
                        if 'override' in self.voronoi_thres.keys():
                            for item in self.voronoi_thres['override']:
                                for cls in item[0]:
                                    pos_thres[cls] = item[1][0]
                                    neg_thres[cls] = item[1][1]
                        loss_bbox_vor += self.loss_voronoi((mu, sigma.bmm(sigma)),
                                                        label, self.images[batch_id],
                                                        pos_thres, neg_thres,
                                                        voronoi=self.voronoi_type)
                        self.vis[batch_id] = self.loss_voronoi.vis

             # Aggregate targets of the same instance based on their identical bid
            bid_with_view = pos_bid_targets[:, 3] + 0.5 * pos_bid_targets[:, 2]
            bid, idx = torch.unique(bid_with_view, return_inverse=True)

            # Generate a mask to eliminate bboxes without correspondence
            # (bcnt is supposed to be 3, for ori, rot, and flp)
            ins_bid_with_view = bid.new_zeros(*bid.shape).index_reduce_(
                0, idx, bid_with_view, 'amin', include_self=False)
            _, bidx, bcnt = torch.unique(
                ins_bid_with_view.long(),
                return_inverse=True,
                return_counts=True)
            bmsk = bcnt[bidx] == 2

            ins_bids = pos_bid_targets.new_zeros(*bid.shape).index_reduce_(
                    0, idx, pos_bid_targets[:, 3], 'amin', include_self=False)
            
            ins_batch = pos_bid_targets.new_zeros(*bid.shape).index_reduce_(
                    0, idx, pos_bid_targets[:, 0], 'amin', include_self=False)
            
            ins_labels = pos_labels.new_zeros(*bid.shape).index_reduce_(
                    0, idx, pos_labels, 'amin', include_self=False)
            
             # Use gt point to replace predicted center for other losses
            pos_rbox_preds = torch.cat((pos_rbox_targets[:, :2], 
                                        pos_rbox_preds[:, 2:]), -1)
            ins_rbox_preds = pos_rbox_preds.new_zeros(
                *bid.shape, pos_rbox_preds.shape[-1]).index_reduce_(
                    0, idx, pos_rbox_preds, 'mean',
                    include_self=False)
            
            #ins_rbox_targets = pos_rbox_targets.new_zeros(
            #    *bid.shape, pos_rbox_targets.shape[-1]).index_reduce_(
            #        0, idx, pos_rbox_targets, 'mean',
            #        include_self=False)
            #  Batched RBox for Edge Loss
            batched_rbox = []
            if self.iter_count >= self.edge_loss_start_iter:
                for batch_id in range(len(batch_gt_instances)):
                    group_mask = (ins_batch == batch_id) & (ins_bids != 0)
                    rbox = ins_rbox_preds[group_mask]
                    label = ins_labels[group_mask]
                    edge_loss_mask = torch.zeros_like(label, dtype=torch.bool)
                    for c in self.edge_loss_cls:
                        edge_loss_mask = torch.logical_or(edge_loss_mask, label == c)
                    batched_rbox.append(rbox[edge_loss_mask])
                loss_bbox_edg = self.loss_bbox_edg(batched_rbox, self.edges)

            loss_bbox_ovl = loss_bbox_ovl / len(batch_gt_instances)
            loss_bbox_vor = loss_bbox_vor / len(batch_gt_instances)
            loss_bbox_edg = loss_bbox_edg / len(batch_gt_instances)

            
            pair_labels = ins_labels[bmsk].view(-1, 2)[:, 0]
            square_mask = torch.zeros_like(pair_labels, dtype=torch.bool)
            for c in self.square_cls:
                square_mask = torch.logical_or(square_mask, pair_labels == c)

            pair_cls_scores = torch.empty(
                *bid.shape, device=bid.device).index_reduce_(
                0, idx, pos_cls_scores, 'mean',
                include_self=False)[bmsk].view(-1, 2)

            pair_angle_preds = torch.empty(
                *bid.shape, pos_angle_preds.shape[-1],
                device=bid.device).index_reduce_(
                0, idx, pos_angle_preds, 'mean',
                include_self=False)[bmsk].view(-1, 2,
                                               pos_angle_preds.shape[-1])
            pair_angle_preds = self.angle_coder.decode(
                pair_angle_preds, keepdim=True)
            
            pair_rbox_preds = torch.empty(
                *bid.shape, pos_rbox_preds.shape[-1], 
                device=bid.device).index_reduce_(
                0, idx, pos_rbox_preds, 'mean',
                include_self=False)[bmsk].view(-1, 2, 5)
            

            # Self-supervision
            ss_info = batch_img_metas[0]['ss']
            valid = pair_cls_scores[:, 1] > 0.1
            
            bbox_area = pair_rbox_preds[:, 0, 2] * pair_rbox_preds[:, 0, 3]
            sca = ss_info[1] if ss_info[0] == 'sca' else 1
            valid = torch.logical_and(valid, bbox_area > 0)
            valid = torch.logical_and(valid, bbox_area * sca > 0)
            valid = torch.logical_and(valid, bbox_area < INF)
            valid = torch.logical_and(valid, bbox_area * sca < INF)

            if torch.any(valid):
                if num_p :
                    if torch.any(valid_p):
                        ori_preds = pair_gaus_preds[valid_p, 0]
                        trs_preds = pair_gaus_preds[valid_p, 1]
                    
                square_mask = square_mask[valid]
                ori_angle = pair_angle_preds[valid, 0]
                trs_angle = pair_angle_preds[valid, 1]

                if ss_info[0] == 'rot':
                    rot = ori_angle.new_tensor(ss_info[1])
                    cos_r = torch.cos(rot)
                    sin_r = torch.sin(rot)
                    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
                    
                    if num_p:
                        if torch.any(valid_p):
                            ori_preds = R.matmul(ori_preds).matmul(R.permute(0, 2, 1))
                            loss_ss = gwd_loss((None, ori_preds.bmm(ori_preds)), (None, trs_preds.bmm(trs_preds)))
                        else:
                            loss_ss = pos_bbox_preds.new_tensor(0)
                    else:
                        loss_ss = pos_bbox_preds.new_tensor(0)
                    d_ang = trs_angle - ori_angle - ss_info[1]
                    d_ang = (d_ang + math.pi / 2) % math.pi - math.pi / 2
                    d_ang[square_mask] = 0
                    loss_ssa = self.loss_ss(d_ang, torch.zeros_like(d_ang))
                elif ss_info[0] == 'flp':
                    
                    if num_p:
                        if torch.any(valid_p):
                            ori_preds = ori_preds * ori_preds.new_tensor((1, -1, -1, 1)).reshape(2, 2)
                            loss_ss = gwd_loss((None, ori_preds.bmm(ori_preds)), (None, trs_preds.bmm(trs_preds)))
                        else:
                            loss_ss = pos_bbox_preds.new_tensor(0)
                    else:
                        loss_ss = pos_bbox_preds.new_tensor(0)
                    d_ang = trs_angle + ori_angle
                    d_ang = (d_ang + math.pi / 2) % math.pi - math.pi / 2
                    d_ang[square_mask] = 0
                    loss_ssa = self.loss_ss(d_ang, torch.zeros_like(d_ang))
                else:
                    
                    if num_p:
                        if torch.any(valid_p):
                            sca = ori_preds.new_tensor(ss_info[1])
                            ori_preds = ori_preds * sca
                            loss_ss = gwd_loss((None, ori_preds.bmm(ori_preds)), (None, trs_preds.bmm(trs_preds)))
                        else:
                            loss_ss = pos_bbox_preds.new_tensor(0)
                    else:
                        loss_ss = pos_bbox_preds.new_tensor(0)
                    d_ang = trs_angle - ori_angle
                    d_ang = (d_ang + math.pi / 2) % math.pi - math.pi / 2
                    d_ang[square_mask] = 0
                    loss_ssa = self.loss_ss(d_ang, torch.zeros_like(d_ang))
                loss_ss = self.gwd_weight * loss_ss
            else:
                loss_ss = pos_bbox_preds.new_tensor(0)
                loss_ssa = 0 * pos_angle_preds.sum()
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_bbox_vor = pos_bbox_preds.sum()
            loss_bbox_ovl = pos_bbox_preds.sum()
            loss_bbox_edg = pos_bbox_preds.sum()
            loss_ss = pos_bbox_preds.sum()
            loss_ssa = pos_angle_preds.sum()
            loss_hbox = pos_bbox_preds.sum()
            loss_cent = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_bbox_vor=loss_bbox_vor,
            loss_bbox_ovl=loss_bbox_ovl,
            loss_bbox_edg=loss_bbox_edg,
            loss_ss=loss_ss,
            loss_ssa=loss_ssa,
            loss_hbox=loss_hbox,
            loss_cent=loss_cent,
            loss_centerness = loss_centerness
        )

    def get_targets(self, points, batch_gt_instances, strides, use_single=True):
        """Compute regression, classification and centerness targets for points
        in multiple images.
        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
        Returns:
            tuple: Targets of each level.
            - concat_lvl_labels (list[Tensor]): Labels of each level.
            - concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                level.
            - concat_lvl_angle_targets (list[Tensor]): Angle targets of \
                each level.
        """
        
        regress_ranges = self.regress_ranges
       

        num_levels = len(points)
        
        assert len(points) == len(regress_ranges)


        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        if use_single:
            labels_list, bbox_targets_list, angle_targets_list, \
            bid_targets_list, centerness_targets_list =  multi_apply(
                                    self._get_targets_single_point,
                                    batch_gt_instances,
                                    points=concat_points,
                                    regress_ranges=concat_regress_ranges,
                                    num_points_per_lvl=num_points)
        else:    
            labels_list, bbox_targets_list, angle_targets_list, \
                bid_targets_list, centerness_targets_list = multi_apply(
                self._get_targets_single_hr,
                batch_gt_instances,
                points=concat_points,
                regress_ranges=concat_regress_ranges,
                num_points_per_lvl=num_points)

        # split to per img, per level

        
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
       
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        angle_targets_list = [
            angle_targets.split(num_points, 0)
            for angle_targets in angle_targets_list
        ]
        bid_targets_list = [
            bid_targets.split(num_points, 0)
            for bid_targets in bid_targets_list
        ]
        centerness_targets_list = [centerness_targets.split(num_points, 0)
                                   for centerness_targets in centerness_targets_list]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_angle_targets = []
        concat_lvl_bid_targets = []
        concat_lvl_centerness_targets = []

        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            angle_targets = torch.cat(
                [angle_targets[i] for angle_targets in angle_targets_list])
            bid_targets = torch.cat(
                [bid_targets[i] for bid_targets in bid_targets_list])
            centerness_targets = torch.cat(
                [centerness_targets[i] for centerness_targets in centerness_targets_list])
            if self.norm_on_bbox:
                if use_single:
                    bbox_targets = bbox_targets / self.strides[i]
                else:
                    bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_centerness_targets.append(centerness_targets)
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_angle_targets.append(angle_targets)
            concat_lvl_bid_targets.append(bid_targets)

        return (concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_angle_targets,
                concat_lvl_bid_targets, concat_lvl_centerness_targets)

    def _get_targets_single_point(
            self, gt_instances, points,
            regress_ranges,
            num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = len(gt_instances['labels'])
        gt_bboxes = gt_instances['bboxes']
        gt_labels = gt_instances['labels']
        gt_bids = gt_instances['bids']

        if num_gts == 0:
            return gt_labels.new_full((num_points,2), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_bboxes.new_zeros((num_points, 1)), \
                   gt_bids.new_zeros((num_points, 4)), \
                   gt_bboxes.new_zeros((num_points, ))

        areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]
        # figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)

        cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle) ######
        rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle],
                               dim=-1).reshape(num_points, num_gts, 2, 2)##########
        
        offset = points - gt_ctr
        offset = torch.matmul(rot_matrix, offset[..., None])######
        offset = offset.squeeze(-1)########

        w, h = gt_wh[..., 0].clone(), gt_wh[..., 1].clone()

        center_r = torch.clamp((w * h).sqrt() / 64, 1, 5)[..., None]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius_p
            stride = offset.new_zeros(offset.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            inside_center_bbox_mask = (abs(offset) < stride * center_r).all(dim=-1)
            inside_gt_bbox_mask = torch.logical_and(inside_center_bbox_mask,
                                                    inside_gt_bbox_mask)
            #inside_gt_bbox_mask = (abs(offset) < stride * center_r).all(dim=-1)

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        #centerness_targets = gt_bboxes.new_zeros((num_points, ))
        centerness_targets = self.centerness_target(bbox_targets)

        angle_targets = gt_angle[range(num_points), min_area_inds]
        bid_targets = gt_bids[min_area_inds]
        #bbox_targets = torch.cat((bbox_targets, angle_targets), -1)

        return labels, bbox_targets, angle_targets, bid_targets, centerness_targets

    def _get_targets_single_hr(self, gt_instances, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression, classification and angle targets for a single
        image."""
       
        gt_bboxes = gt_instances['bboxes']
        gt_labels = gt_instances['labels']
        gt_bids = gt_instances['bids']

        num_points = points.size(0)
        num_gts = len(gt_instances['labels'])
        if num_gts == 0:
            return gt_labels.new_full((num_points,2), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_bboxes.new_zeros((num_points, 1)), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_bboxes.new_zeros((num_points, ))

        areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]
        # figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)

        cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
        rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle],
                               dim=-1).reshape(num_points, num_gts, 2, 2)
        offset = points - gt_ctr
        offset = torch.matmul(rot_matrix, offset[..., None])
        offset = offset.squeeze(-1)

        w, h = gt_wh[..., 0], gt_wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            stride = offset.new_zeros(offset.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            inside_center_bbox_mask = (abs(offset) < stride).all(dim=-1)
            inside_gt_bbox_mask = torch.logical_and(inside_center_bbox_mask,
                                                    inside_gt_bbox_mask)
        else:
            gaussian_center = offset_x.pow(2) / (w / 2).pow(2) + offset_y.pow(2) / (h / 2).pow(2)
            inside_gt_bbox_mask = gaussian_center < 1


        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        if self.center_sampling:
            centerness_targets = self.centerness_target(bbox_targets)
        else:
            centerness_targets = 1 - gaussian_center[range(num_points), min_area_inds]

        angle_targets = gt_angle[range(num_points), min_area_inds]
        bid_targets = gt_bids[min_area_inds]
        #bbox_targets = torch.cat((bbox_targets, angle_targets), -1)

        

        return labels, bbox_targets, angle_targets, bid_targets, centerness_targets

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)
        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                 left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                 top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   angle_preds,
                   centernesses,
                   img_metas,
                   cfg=None,
                   rescale=None,
                   batch_gt_instances=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W)
            angle_preds (list[Tensor]): Box angle for each scale level \
                with shape (N, num_points * 1, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 6) tensor, where the first 5 columns
                are bounding box positions (x, y, w, h, angle) and the 6-th
                column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        
        assert len(cls_scores) == len(bbox_preds) ==len(angle_preds) == len(centernesses)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            angle_pred_list = [
                angle_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            if self.training:
                mlvl_points = self.prior_generator_single.grid_priors(featmap_sizes,
                                                       bbox_preds[0].dtype,
                                                       bbox_preds[0].device)
                det_bboxes = self._get_bboxes_single_pseudo(cls_score_list,
                                                            bbox_pred_list,
                                                            angle_pred_list,
                                                            mlvl_points, img_shape,
                                                            scale_factor, cfg, rescale, batch_gt_instances[img_id])
            else:
                mlvl_points = self.prior_generator.grid_priors(featmap_sizes,
                                                       bbox_preds[0].dtype,
                                                       bbox_preds[0].device)
                
                det_bboxes = self._get_bboxes_single(cls_score_list,
                                                     bbox_pred_list,
                                                     angle_pred_list,
                                                     centerness_pred_list,
                                                     mlvl_points, img_shape,
                                                     scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single_pseudo(self,
                                  cls_scores,
                                  bbox_preds,
                                  angle_preds,
                                  mlvl_points,
                                  img_shape,
                                  scale_factor,
                                  cfg,
                                  rescale=False,
                                  batch_gt_instances=None):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * 4, H, W).
            angle_preds (list[Tensor]): Box angle for a single scale level \
                with shape (N, num_points * 1, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 6), where the first 5 columns
                are bounding box positions (x, y, w, h, angle) and the
                6-th column is a score between 0 and 1.
        """
        gt_instances = batch_gt_instances
        if self.training:
            scale_factor = [1, 1]
        else:
            scale_factor = scale_factor
        gt_bboxes = gt_instances['bboxes']
        gt_labels = gt_instances['labels']
        # 在特征图上的位置
        gt_pos = (gt_bboxes[:, 0:2] / self.strides[0] * scale_factor[1]).long()  # num_gt*2


        cls_score, bbox_pred, angle_pred = cls_scores[0], bbox_preds[0], angle_preds[0]
        
        H, W = cls_score.shape[1:3]

        # num_gt * 1
        gt_valid_mask = (0 <= gt_pos[:, 0]) & (gt_pos[:, 0] < W) & (0 <= gt_pos[:, 1]) & (gt_pos[:, 1] < H)
        gt_idx = gt_pos[:, 1] * W + gt_pos[:, 0]
        gt_idx = gt_idx.clamp(0, cls_score[0].numel() - 1)
        # 提取特征图上的预测值
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)[gt_idx]

        cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)[gt_idx]

        angle_pred = angle_pred.permute(1, 2, 0).reshape(
            -1, self.angle_coder.encode_size)[gt_idx]
        decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)

        #bboxes = torch.cat((gt_bboxes[:, 0:2], bbox_pred[:, :2] * 2, decoded_angle), -1)
        points = mlvl_points[0][gt_idx]
        bbox_pred_decode = self.bbox_coder.decode(
            points, torch.cat((bbox_pred, decoded_angle), -1))#, max_shape=img_shape)
        bboxes = torch.cat((gt_bboxes[:, 0:2], bbox_pred_decode[:, 2:]), -1)

        bboxes[~gt_valid_mask, 2:] = 0
        bboxes[:, 2:4][bboxes[:, 2:4] < 24] = 24
        bboxes[:, 2:4] = bboxes[:, 2:4] / scale_factor[1]

        # Replace point labels and keep RBox/HBox as is
        assert gt_labels.dim() == 2
        gt_point = gt_labels[:, 1] == 2
        bboxes[~gt_point, 2:] = gt_bboxes[~gt_point, 2:]

        for id in self.post_process.keys():
            bboxes[gt_labels[:, 0] == id, 2:4] *= self.post_process[id]
        for id in self.square_cls:
            bboxes[gt_labels[:, 0] == id, -1] = 0

        bboxes = torch.cat((bboxes, torch.ones_like(bboxes[:, :1])), dim=1)

        return bboxes, gt_labels

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           angle_preds,
                           centernesses,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           batch_gt_instances=None):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * 4, H, W).
            angle_preds (list[Tensor]): Box angle for a single scale level \
                with shape (N, num_points * 1, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 6), where the first 5 columns
                are bounding box positions (x, y, w, h, angle) and the
                6-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, angle_pred, centerness, points in zip(
                cls_scores, bbox_preds, angle_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(-1, self.angle_coder.encode_size)
            angle_pred = self.angle_coder.decode(angle_pred, keepdim=True)
            bbox_pred = torch.cat([bbox_pred, angle_pred], dim=1)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * 1).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                #angle_pred = angle_pred[topk_inds, :]  # add
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            # bboxes = torch.cat((points + bbox_pred[:, 2:], bbox_pred[:, :2] * 2, angle_pred), -1)   # add

            bboxes = self.bbox_coder.decode(
                points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            scale_factor = mlvl_bboxes.new_tensor(scale_factor)
            mlvl_bboxes[..., :4] = mlvl_bboxes[..., :4] / scale_factor
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_centerness = torch.cat(mlvl_centerness)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        det_bboxes, det_labels = multiclass_nms_rotated(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness)
        for id in self.square_cls:
            det_bboxes[det_labels == id, 4] = 0
        return det_bboxes, det_labels

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds'))
    def refine_bboxes(self, cls_scores, bbox_preds, angle_preds):
        """This function will be used in S2ANet, whose num_anchors=1."""
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)
        num_imgs = cls_scores[0].size(0)
        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        # device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_points = self.prior_generator.grid_priors(featmap_sizes,
                                                       bbox_preds[0].dtype,
                                                       bbox_preds[0].device)
        bboxes_list = [[] for _ in range(num_imgs)]

        for lvl in range(num_levels):
            bbox_pred = bbox_preds[lvl]
            angle_pred = angle_preds[lvl]
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(num_imgs, -1, 4)
            angle_pred = angle_pred.permute(0, 2, 3, 1)
            angle_pred = angle_pred.reshape(num_imgs, -1, 1)
            bbox_pred = torch.cat([bbox_pred, angle_pred], dim=-1)

            points = mlvl_points[lvl]

            for img_id in range(num_imgs):
                bbox_pred_i = bbox_pred[img_id]
                decode_bbox_i = self.bbox_coder.decode(points, bbox_pred_i)
                bboxes_list[img_id].append(decode_bbox_i.detach())

        return bboxes_list