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
class Semi_GmmLossforLabeledData(nn.Module):
    def __init__(self, 
                 cls_channels=len(CLASSES),
                 loss_type='origin', 
                 bbox_loss_type='l1', 
                 image_class_prompt_path='/mnt/nas-new/home/zhanggefan/liuxiang/mcl/Assinger_Assistent/image_class_prompt_from_chat_with_percent10_label_modified.pt',
                 policy = 'high',
                 angle_coder=dict(
                     type='PSCCoder',
                     angle_version='le90',
                     dual_freq=False,
                     num_step=3,
                     thr_mod=0),
                strides=[8, 16, 32, 64, 128]):
        super(Semi_GmmLossforLabeledData, self).__init__()
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


    def loss_single(self, t_logits_list, s_logits_list, level_inds, ratio=0.03, img_metas=None, bbox_head=None, featmap_sizes=None):
        t_cls_scores, t_bbox_preds, t_centernesses = tuple(t_logits_list)
        s_cls_scores, s_bbox_preds, s_centernesses = tuple(s_logits_list)

        batch_gt_instances = []
        batch_gt_instance = {}
        for i in range(len(img_metas['img'])):
            batch_gt_instance['bboxes'] = img_metas['gt_bboxes'][i].to(img_metas['gt_bboxes'][i].device)
            original_labels = img_metas['gt_labels'][i]
            batch_gt_instance['labels'] \
                 = torch.stack((original_labels, torch.ones_like(original_labels)), dim=1).to(img_metas['gt_labels'][i].device)
            batch_gt_instance['bids'] = torch.zeros(len(img_metas['gt_labels'][0]), 4).to(img_metas['gt_labels'][i].device)
            batch_gt_instances.append(batch_gt_instance)

        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=s_bbox_preds.dtype,
            device=s_bbox_preds.device)
        
        labels, bbox_targets, angle_targets, bid_targets, centerness_targets = bbox_head.get_targets(
            all_level_points, batch_gt_instances, self.strides)
        

        bg_class_ind = len(CLASSES) # 15 表示背景
        flatten_labels = torch.cat(labels)
        flatten_labels = flatten_labels[:, 0]
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1) 

        with torch.no_grad():

            teacher_probs = t_cls_scores.sigmoid()
            teacher_bboxes = t_bbox_preds
            teacher_centernesses = t_centernesses.sigmoid()

            # P3
            teacher_probs_p3 = teacher_probs[:level_inds[0]]
            teacher_centernesses_p3 = teacher_centernesses[:level_inds[0]]
            joint_confidences_p3 = teacher_probs_p3 * teacher_centernesses_p3
            max_vals_p3 = torch.max(joint_confidences_p3, 1)[0]
            selected_inds_p3 = torch.topk(max_vals_p3, joint_confidences_p3.size(0))[1][:2000]

            # P4
            teacher_probs_p4 = teacher_probs[level_inds[0]:level_inds[1]]
            teacher_centernesses_p4 = teacher_centernesses[level_inds[0]:level_inds[1]]
            joint_confidences_p4 = teacher_probs_p4 * teacher_centernesses_p4
            select_inds_p4 = torch.arange(level_inds[0], level_inds[1]).to(joint_confidences_p4.device)
            max_vals_p4 = torch.max(joint_confidences_p4, 1)[0]
            selected_inds_p4 = select_inds_p4[torch.topk(max_vals_p4, joint_confidences_p4.size(0))[1][:2000]]

            # P5, P6, P7
            confidences_rest = teacher_probs[level_inds[1]:]
            selected_inds_rest = torch.arange(level_inds[1], teacher_probs.shape[0]).to(confidences_rest.device)\

    
            filename = img_metas['img_metas'][0]['filename'].split('/')[-1]
            exist_classes_tensor = self.class_name_to_index[filename].to(t_cls_scores.device)
            # 确定选取的 pixel 数量
            count_num = int(t_cls_scores.size(0) * ratio)  # 总共需要选取的正样本数量
            joint_confidences_all = torch.cat([joint_confidences_p3, joint_confidences_p4, confidences_rest], 0)
            # 按联合置信度找到最大值及其类别
            max_vals, max_inds = torch.max(joint_confidences_all, dim=1)  # (21824,), (21824,)  

            # ------------------------------------
            # 1. 输入 Prompt 类别最优先选取
            prompt_mask = torch.isin(max_inds, exist_classes_tensor)
            prompt_confidence_mask = max_vals > 0.02
            prompt_mask = prompt_mask & prompt_confidence_mask
            prompt_vals = max_vals[prompt_mask]
            prompt_inds = torch.where(prompt_mask)[0]

            # 选取 Prompt 类别中 TopK
            prompt_count = count_num * 0.03  # Prompt 类别选取的数量
            if len(prompt_inds) > prompt_count:
                sorted_vals, sorted_inds = torch.topk(prompt_vals, prompt_count)
                prompt_inds = prompt_inds[sorted_inds]  # 最终选取的 Prompt 索引


            #print(f"Prompt Count for labeled data: {prompt_count}")


            '''
            # ------------------------------------
            # 2. 整体 TopK
            overall_count = count_num - prompt_count  # 剩余需要选取的数量
            if overall_count > 0:
                non_prompt_mask = ~prompt_mask  # 非 Prompt 类别的掩码
                non_prompt_vals = max_vals[non_prompt_mask]
                non_prompt_inds = torch.where(non_prompt_mask)[0]

                sorted_vals, sorted_inds = torch.topk(non_prompt_vals, overall_count)
                overall_inds = non_prompt_inds[sorted_inds]  # 整体 TopK 的索引
            else:
                overall_inds = torch.tensor([], device=t_cls_scores.device, dtype=torch.long)
            '''  
            # ------------------------------------
            # 3. 类别 TopK（Prompt 以外）
            non_prompt_classes = torch.arange(t_cls_scores.size(1), device=t_cls_scores.device)
            non_prompt_classes = non_prompt_classes[~torch.isin(non_prompt_classes, exist_classes_tensor)]  # 非 Prompt 类别

            #class_topk_count = int(count_num * 0.001)
            class_topk_count = 1
            class_topk_inds = []

            for cls in non_prompt_classes:
                cls_mask = (max_inds == cls)
                cls_vals = max_vals[cls_mask]
                cls_inds = torch.where(cls_mask)[0]

                if len(cls_inds) >= class_topk_count:
                    sorted_vals, sorted_inds = torch.topk(cls_vals, class_topk_count)
                    cls_inds = cls_inds[sorted_inds]  # 选取 TopK
                class_topk_inds.append(cls_inds)

            class_topk_inds = torch.cat(class_topk_inds) if class_topk_inds else torch.tensor([], device=t_cls_scores.device)
            #print(f"Class TopK Count for labeled data: {len(class_topk_inds)}")

            # coarse_inds
            selected_inds_coarse = torch.cat([selected_inds_p3, selected_inds_p4, selected_inds_rest], 0)
            selected_inds_coarse = torch.unique(selected_inds_coarse)

            # fine_inds
            self.mean_score = max_vals.mean()
            thres = self.gmm_policy(max_vals, policy = self.policy)
            self.thres = thres
            selected_inds = torch.nonzero(max_vals >thres).squeeze(-1)

            selected_inds, counts = torch.cat([selected_inds_coarse, selected_inds], 0).unique(return_counts=True)
            selected_inds = selected_inds[counts>1]

            
            selected_inds = torch.cat([prompt_inds, class_topk_inds, selected_inds, pos_inds], 0).unique().long()
            #print(f"Selected Indices Count for labeled data: {len(selected_inds)}")
            

            weight_mask = torch.zeros_like(max_vals)
            weight_mask[selected_inds] = max_vals[selected_inds]
            b_mask = weight_mask > 0.

            is_from_class_topk = torch.isin(selected_inds, class_topk_inds)

            # 初始化全零的权重 mask
            weight_mask_class_topk = torch.zeros_like(max_vals)
            weight_mask_others = torch.zeros_like(max_vals)

            if is_from_class_topk.any():
                inds_topk = selected_inds[is_from_class_topk]
                inds_other = selected_inds[~is_from_class_topk]

                weight_mask_class_topk[inds_topk] = 1.0  # 用1.0表示正样本 presence
                weight_mask_others[inds_other] = max_vals[inds_other]
            else:
                # fallback 情况：全部作为 other
                weight_mask_others[selected_inds] = max_vals[selected_inds]

            b_mask_class_topk = weight_mask_class_topk > 0.
            b_mask_others = weight_mask_others > 0.


        if (b_mask.sum() == 0) or (b_mask_others.sum() == 0):
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
                    weight=weight_mask_others,#使用样本对应的置信度进行加权
                    reduction="sum",
                ) / weight_mask_others.sum()
            if is_from_class_topk.any():
                loss_cls_class_topk = QFLv2(
                    s_cls_scores.sigmoid(),
                    teacher_probs,
                    weight=weight_mask_class_topk, 
                    reduction="sum",
                ) / (b_mask_class_topk * max_vals).sum() 
                loss_cls = loss_cls + loss_cls_class_topk
            #cls的loss对前景和背景都计算loss，bbox和中心度只对正样本计算loss
            if self.bbox_loss_type == 'l1':
                loss_bbox = (self.bbox_loss(
                    s_bbox_preds[b_mask_others],
                    teacher_bboxes[b_mask_others],
                ) * weight_mask_others[:, None][b_mask_others]).mean() * 10
                if is_from_class_topk.any():
                    loss_bbox_class_topk = (self.bbox_loss(
                        s_bbox_preds[b_mask_class_topk],
                        teacher_bboxes[b_mask_class_topk],
                        ) * t_centernesses.sigmoid()[b_mask_class_topk]).mean()
                    loss_bbox = loss_bbox + loss_bbox_class_topk
            else:
                all_level_points = self.prior_generator.grid_priors(
                    featmap_sizes,
                    dtype=s_bbox_preds.dtype,
                    device=s_bbox_preds.device)
                flatten_points = torch.cat(all_level_points)
                s_bbox_preds = self.bbox_coder.decode(flatten_points, s_bbox_preds)[b_mask_others]
                t_bbox_preds = self.bbox_coder.decode(flatten_points, t_bbox_preds)[b_mask_others]
                loss_bbox = self.bbox_loss(
                    s_bbox_preds,
                    t_bbox_preds,
                ) * t_centernesses.sigmoid()[b_mask_others]
                if is_from_class_topk.any():
                    loss_bbox_class_topk = self.bbox_loss(
                        s_bbox_preds,
                        t_bbox_preds,
                    ) * t_centernesses.sigmoid()[b_mask_class_topk]
                    loss_bbox = loss_bbox + loss_bbox_class_topk

                nan_indexes = ~torch.isnan(loss_bbox)
                if nan_indexes.sum() == 0:
                    loss_bbox = torch.zeros(1, device=s_cls_scores.device).sum()
                else:
                    loss_bbox = loss_bbox[nan_indexes].mean()


            loss_centerness = (F.binary_cross_entropy(
                s_centernesses[b_mask_others].sigmoid(),
                teacher_centernesses[b_mask_others],
                reduction='none'
            )* weight_mask[:, None][b_mask_others]).mean() * 10
            if is_from_class_topk.any():
                loss_centerness_class_topk = F.binary_cross_entropy(
                    s_centernesses[b_mask_class_topk].sigmoid(),
                    t_centernesses[b_mask_class_topk].sigmoid(),
                    reduction='mean'
                )
                loss_centerness = loss_centerness + loss_centerness_class_topk

        return loss_cls, loss_bbox, loss_centerness

    def forward(self, teacher_logits, student_logits, ratio=0.03, img_metas=None, bbox_head=None, **kwargs):

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
                            bbox_head=bbox_head,
                            featmap_sizes=featmap_sizes)

        unsup_losses = dict(
            loss_cls=sum(losses_list[0]) / len(losses_list[0]),
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