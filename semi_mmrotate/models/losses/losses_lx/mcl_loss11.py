from contextlib import nullcontext
from numpy.lib.function_base import select
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import multi_apply
from mmrotate.models import ROTATED_LOSSES
from mmrotate.core import build_bbox_coder
import numpy as np
import sklearn.mixture as skm
from mmdet.core.anchor.point_generator import MlvlPointGenerator
from mmrotate.models import ROTATED_LOSSES, build_loss

INF = 1e8
CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
            'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
            'basketball-court', 'storage-tank', 'soccer-ball-field',
            'roundabout', 'harbor', 'swimming-pool', 'helicopter',
            )


@ROTATED_LOSSES.register_module()
class Semi_GmmLoss11(nn.Module):
    def __init__(self, 
                 cls_channels=len(CLASSES),
                 loss_type='origin', 
                 bbox_loss_type='l1', 
                 image_class_prompt_path='/mnt/nas-new/home/zhanggefan/liuxiang/mcl/Assinger_Assistent/image_class_prompt_from_chat.pt',
                 policy = 'high',
                 angle_coder=dict(
                     type='PSCCoder',
                     angle_version='le90',
                     dual_freq=False,
                     num_step=3,
                     thr_mod=0),
                 strides=[8, 16, 32, 64, 128]):
        super(Semi_GmmLoss11, self).__init__()
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

        self.policy = policy
        self.angle_coder = build_bbox_coder(angle_coder)
        self.strides = strides 
        self.mean_score = []
        self.thres = []
    '''
    将“以层级为中心、保留空间结构”的数据，转换为“以图像为中心、扁平化”的列表结构
    这样可以方便后续逐个位置的loss计算、以及筛选预测等操作。
    '''
    def pre_processing(self, logits):

        cls_scores, bbox_preds, angle_preds, centernesses = logits
        assert len(cls_scores) == len(bbox_preds) == len(angle_preds) == len(centernesses)

        batch_size = cls_scores[0].shape[0]   

        decode_angle_preds = []
        for angle_pred in angle_preds:
           single_lvl = []
           for i in range(batch_size):
               flatten_angle_pred = angle_pred[i].permute(1, 2, 0).reshape(-1, self.angle_coder.encode_size)
               de_angle_pred = self.angle_coder.decode(
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


            img_logits.append(torch.cat([
                torch.cat([x[img_id].permute(1, 2, 0).reshape(-1, 4), y[img_id]], dim=-1) for x, y in
                zip(bbox_preds, decode_angle_preds)
            ], dim=0))


            img_logits.append(torch.cat([
                x[img_id].permute(1, 2, 0).reshape(-1, 1) for x in centernesses
            ], dim=0))

            img_logits_list.append(img_logits)
        #img_logits_list里面元素为img_lofits，数量为batch_size，每个元素代表每张图像的预测结果，包含类别分数、边界框预测和中心度预测
        #img_logits里面每个元素包含了对应图片的所有FPN层的预测结果
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

    def select_topk_gmm(self, start_idx, end_idx, use_centerness, teacher_probs, teacher_centernesses, mask, topk_num=2000):
        topk_num = int(topk_num)
        layer_probs = teacher_probs[start_idx:end_idx]
        if use_centerness:
            layer_centernesses = teacher_centernesses[start_idx:end_idx]
            layer_confidences = layer_probs * layer_centernesses
        else:
            layer_confidences = layer_probs

        max_vals, _ = torch.max(layer_confidences, 1)
        layer_mask = mask[start_idx:end_idx]
        
        layer_vals = max_vals[layer_mask]
        layer_inds = torch.where(layer_mask)[0] + start_idx

        selected_indices_coarse = torch.empty(0, dtype=torch.long, device=teacher_probs.device)
        if layer_mask.any():
            topk_num = min(topk_num, layer_vals.size(0))
            if topk_num > 0:
                _, topk_indices = torch.topk(layer_vals, topk_num)
                selected_indices_coarse = layer_inds[topk_indices]
                
        self.mean_score.append(layer_vals.mean())
        thres = self.gmm_policy(layer_vals, policy=self.policy)
        self.thres.append(thres)

        fine_mask = layer_vals > thres
        selected_inds_fine = layer_inds[fine_mask]
        selected_indices, counts = torch.cat([selected_indices_coarse, selected_inds_fine], dim=0).unique(return_counts=True)
        selected_indices = selected_indices.long()
        selected_indices = selected_indices[counts > 1]
        

        return selected_indices
    
    def update_prompt_mask(self, teacher_probs, prompt_mask, exist_classes_tensor, threshold=1.2):
        """
        根据分类分数和 prompt 类别索引张量更新 prompt 掩码。

        参数:
        - teacher_probs (torch.Tensor): 样本的分类分数张量，形状为 (样本数, 类别数)。
        - prompt_mask (torch.Tensor): 布尔类型的 prompt 掩码，形状为 (样本数, )。
        - prompt_indices_tensor (torch.Tensor): 一个一维数值张量，包含了所有 prompt 类别的索引。
                                            例如 torch.tensor([0, 1, 2])。
                                            推荐使用 torch.long 类型。
        - threshold (float): 分数差值的阈值。

        返回:
        - torch.Tensor: 更新后的 prompt 掩码。
        """
        # 1. 筛选出当前不在 prompt 掩码内的样本点的索引
        outside_mask_indices = torch.where(~prompt_mask)[0]
        
        # 检查索引张量是否为空
        if len(outside_mask_indices) == 0 or exist_classes_tensor.numel() == 0:
        # 返回原始掩码，以及两个空的张量，以匹配输出签名
            empty_indices = torch.tensor([], device=prompt_mask.device, dtype=torch.long)
            return prompt_mask.clone(), empty_indices, empty_indices

        probs_outside_mask = teacher_probs[outside_mask_indices]

        # 2. 计算每个样本的全局最高分
        overall_max_probs, _ = torch.max(probs_outside_mask, dim=1)
        


        # 3. 直接使用索引张量来筛选出 prompt 类的分数
        #    PyTorch 的高级索引功能允许我们传入一个张量作为列索引
        prompt_probs = probs_outside_mask[:, exist_classes_tensor]
        
        # 4. 计算在这些 prompt 类别中的最高分
        max_prompt_probs, local_max_indices = torch.max(prompt_probs, dim=1)
        corresponding_global_classes = exist_classes_tensor[local_max_indices]
        
        # 5. 计算差值并更新掩码
        prob_diff = overall_max_probs / max_prompt_probs
        samples_to_add_mask = prob_diff <= threshold
        
        original_indices_to_add = outside_mask_indices[samples_to_add_mask]
        class_ts_inds = corresponding_global_classes[samples_to_add_mask]

        updated_mask = prompt_mask.clone()
        updated_mask[original_indices_to_add] = True

        return updated_mask, original_indices_to_add, class_ts_inds
    




    def loss_single(self, t_logits_list, s_logits_list, level_inds, ratio=0.03, img_metas=None, featmap_sizes=None):
        t_cls_scores, t_bbox_preds, t_centernesses = tuple(t_logits_list)
        s_cls_scores, s_bbox_preds, s_centernesses = tuple(s_logits_list)

        all_level_points = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=s_bbox_preds.dtype,
                device=s_bbox_preds.device)

        with torch.no_grad():

            teacher_probs = t_cls_scores.sigmoid()
            teacher_bboxes = t_bbox_preds
            teacher_centernesses = t_centernesses.sigmoid()
            joint_confidence_all = teacher_probs * teacher_centernesses
            joint_confidence_all[level_inds[1]:] = teacher_probs[level_inds[1]:] 
            max_vals, max_inds = torch.max(joint_confidence_all, 1)

            #prompt的粗筛和精筛
            filename = img_metas['img_metas'][0]['filename'].split('/')[-1]
            exist_classes_tensor = self.class_name_to_index[filename].to(t_cls_scores.device)
            _, max_inds = torch.max(teacher_probs, 1)
            prompt_mask = torch.isin(max_inds, exist_classes_tensor)
            prompt_mask, original_indices_to_add, max_class_ts= self.update_prompt_mask(teacher_probs, prompt_mask, exist_classes_tensor, threshold=1.2)
            

            layer_configs = [
            (0, level_inds[0], True, level_inds[0] * 0.12),  # P3
            (level_inds[0], level_inds[1], True, (level_inds[1] - level_inds[0]) * 0.24),  # P4
            (level_inds[1], teacher_probs.size(0), False, (teacher_probs.size(0) - level_inds[1]))  # P5+
            ]
            
            selected_prompt_indices = torch.empty(0, device=teacher_probs.device)
            for start_idx, end_idx, use_centerness, topk_num in layer_configs:
                selected_prompt_indices_layers = self.select_topk_gmm(start_idx, end_idx, use_centerness, teacher_probs, 
                                                          teacher_centernesses, prompt_mask, topk_num=topk_num)
                
                selected_prompt_vals_layers = max_vals[selected_prompt_indices_layers]
                thres_prompt = self.gmm_policy(selected_prompt_vals_layers, policy=self.policy)
                mask = selected_prompt_vals_layers > thres_prompt
                selected_prompt_indices_layers = selected_prompt_indices_layers[mask]
    
                selected_prompt_indices = torch.cat([selected_prompt_indices, selected_prompt_indices_layers], dim=0).long()
                
            
            mask_others = torch.ones_like(max_inds, dtype=torch.bool)
            mask_others[selected_prompt_indices] = False 
            selected_indices = torch.empty(0, device=teacher_probs.device)
            for start_idx, end_idx, use_centerness, topk_num in layer_configs:
                selected_indices_layers = self.select_topk_gmm(start_idx, end_idx, use_centerness, teacher_probs, 
                                                          teacher_centernesses, mask_others, topk_num=topk_num)
                
                selected_vals_layers = max_vals[selected_indices_layers]
                thres = self.gmm_policy(selected_vals_layers, policy=self.policy)
                mask = selected_vals_layers > thres
                selected_indices_layers = selected_indices_layers[mask]
                
                selected_indices = torch.cat([selected_indices, selected_indices_layers], dim=0).long()

        
            # 7. 最终的索引
            selected_inds = torch.cat([selected_prompt_indices, selected_indices], 0).unique()  # 确保索引有序
            
            '''
            selected_ts_mask = torch.isin(original_indices_to_add, selected_inds)
            if torch.any(selected_ts_mask):
                selected_ts_inds = original_indices_to_add[selected_ts_mask]
                max_class_origin = max_inds[selected_ts_inds]
                max_class_ts = class_ts_inds[selected_ts_mask]
                # 这一步是关键，它保存了即将被覆盖的数据。
                temp_values = teacher_probs[selected_ts_inds, max_class_origin].clone() # 使用 .clone() 是一个好习惯，避免意外的引用问题

                # 步骤 b: 将 `max_class_ts` 列的值复制到 `max_class_origin` 列的位置。
                # 现在 `max_class_origin` 列的数据已经被覆盖了。
                ts_values = teacher_probs[selected_ts_inds, max_class_ts]
                teacher_probs[selected_ts_inds, max_class_origin] = ts_values

                # 步骤 c: 将存储在临时变量中的原始值复制到 `max_class_ts` 列的位置。
                teacher_probs[selected_ts_inds, max_class_ts] = temp_values
            '''
            max_class_origin = max_inds[original_indices_to_add]
            temp_values = teacher_probs[original_indices_to_add, max_class_origin].clone() # 使用 .clone() 是一个好习惯，避免意外的引用问题

            # 步骤 b: 将 `max_class_ts` 列的值复制到 `max_class_origin` 列的位置。
            # 现在 `max_class_origin` 列的数据已经被覆盖了。
            ts_values = teacher_probs[original_indices_to_add, max_class_ts]
            teacher_probs[original_indices_to_add, max_class_origin] = ts_values

            # 步骤 c: 将存储在临时变量中的原始值复制到 `max_class_ts` 列的位置。
            teacher_probs[original_indices_to_add, max_class_ts] = temp_values
            
            num_prompt_final = torch.tensor(selected_prompt_indices.size(0), device=selected_prompt_indices.device, dtype=torch.long)
            num_final = torch.tensor(selected_inds.size(0), device=selected_inds.device, dtype=torch.long)
            

            joint_confidence_all = teacher_probs * teacher_centernesses
            joint_confidence_all[level_inds[1]:] = teacher_probs[level_inds[1]:] 
            max_vals, _ = torch.max(joint_confidence_all, 1) 
            # 保持 P5+ 层的原始置信度
            weight_mask = torch.zeros_like(max_vals)
            weight_mask[selected_inds] = max_vals[selected_inds]
            b_mask = weight_mask > 0.



        if b_mask.sum() == 0:
            loss_cls = QFLv2(
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
                    weight=weight_mask,#使用样本对应的置信度进行加权
                    reduction="sum",
                ) / weight_mask.sum()

            #cls的loss对前景和背景都计算loss，bbox和中心度只对正样本计算loss
            if self.bbox_loss_type == 'l1':
                loss_bbox = (self.bbox_loss(
                    s_bbox_preds[b_mask],
                    teacher_bboxes[b_mask],
                ) * weight_mask[:, None][b_mask]).mean() * 10

            else:
                all_level_points = self.prior_generator.grid_priors(
                    featmap_sizes,
                    dtype=s_bbox_preds.dtype,
                    device=s_bbox_preds.device)
                flatten_points = torch.cat(all_level_points)
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

            loss_centerness = (F.binary_cross_entropy(
                s_centernesses[b_mask].sigmoid(),
                teacher_centernesses[b_mask],
                reduction='none'
            )* weight_mask[:, None][b_mask]).mean() * 10

        return loss_cls, loss_bbox, loss_centerness, num_final, num_prompt_final

    def forward(self, teacher_logits, student_logits, ratio=0.03, img_metas=None, **kwargs):

        img_t_logits_list = self.pre_processing(teacher_logits)
        img_s_logits_list = self.pre_processing(student_logits)

        featmap_sizes = [featmap.size()[-2:] for featmap in teacher_logits[0]]
        level_inds = []
        start = 0
        for size in featmap_sizes:
            start = start + size[0] * size[0]
            level_inds.append(start)
        level_inds = level_inds[:2]

        # get labels and bbox_targets of each image
        losses_list = multi_apply(
                            self.loss_single,
                            img_t_logits_list,
                            img_s_logits_list,
                            level_inds=level_inds,
                            ratio=ratio,
                            img_metas=img_metas,
                            featmap_sizes=featmap_sizes)

        unsup_losses = dict(
            loss_cls=sum(losses_list[0]) / len(losses_list[0]),
            loss_bbox=sum(losses_list[1]) / len(losses_list[1]),
            loss_centerness=sum(losses_list[2]) / len(losses_list[2]),
            num_final=sum(losses_list[3]) / len(losses_list[3]),
            num_prompt_final=sum(losses_list[4]) / len(losses_list[4]),
        )

        return unsup_losses


def QFLv2(pred_sigmoid,
          teacher_sigmoid,
          weight=None,
          beta=2.0,
          reduction='mean'):
    # all goes to 0 
    pt = pred_sigmoid
    zerolabel = pt.new_zeros(pt.shape) 
    loss = F.binary_cross_entropy(
        pred_sigmoid, zerolabel, reduction='none') * pt.pow(beta) 
    pos = weight > 0

    # positive goes to teacher quality 
    pt = teacher_sigmoid[pos] - pred_sigmoid[pos] 
    loss[pos] = F.binary_cross_entropy(
        pred_sigmoid[pos], teacher_sigmoid[pos], reduction='none') * pt.pow(beta)

    valid = weight >= 0
    if reduction == "mean":
        loss = loss[valid].mean()
    elif reduction == "sum":
        loss = loss[valid].sum()
    return loss