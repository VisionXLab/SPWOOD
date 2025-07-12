import torch
import numpy as np
from .rotated_semi_detector import RotatedSemiDetector
from mmrotate.models.builder import ROTATED_DETECTORS
from mmrotate.models import build_detector


@ROTATED_DETECTORS.register_module()
class MCLTeacher(RotatedSemiDetector):
    def __init__(self, model: dict, semi_loss, att_loss, train_cfg=None, test_cfg=None):
        super(MCLTeacher, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),  # 初始化teacher/student
            semi_loss,
            att_loss,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:  # 训练阶段
            self.freeze("teacher")  # 冻结teacher模型，在训练阶段不参与梯度更新，由EMA更新
            # ugly manner to get start iteration, to fit resume mode
            self.iter_count = train_cfg.get("iter_count", 0)  # 训练迭代步数，用于EMA计算
            # Prepare semi-training config
            # step to start training student (not include EMA update)
            self.burn_in_steps = train_cfg.get("burn_in_steps", 5000)  # 预热阶段，前burn_in_steps次训练只进行监督学习
            # prepare super & un-super weight
            self.sup_weight = train_cfg.get("sup_weight", 1.0)  # 监督损失的权重 
            self.unsup_weight = train_cfg.get("unsup_weight", 1.0)  # 无监督损失的权重
            self.weight_suppress = train_cfg.get("weight_suppress", "linear")
            self.logit_specific_weights = train_cfg.get("logit_specific_weights")

    def forward_train(self, imgs, img_metas, **kwargs):
        super(MCLTeacher, self).forward_train(imgs, img_metas, **kwargs)  # ???
        gt_bboxes = kwargs.get('gt_bboxes')
        gt_labels = kwargs.get('gt_labels')

        # 同步iter_count
        self.teacher.iter_count = self.iter_count
        self.student.iter_count = self.iter_count
        
        # preprocess
        format_data = dict()
        for idx, img_meta in enumerate(img_metas):
            tag = img_meta['tag']
            if tag in ['sup_strong', 'sup_weak']:
                tag = 'sup'
            if tag not in format_data.keys():
                format_data[tag] = dict()
                format_data[tag]['img'] = [imgs[idx]]
                format_data[tag]['img_metas'] = [img_metas[idx]]
                format_data[tag]['gt_bboxes'] = [gt_bboxes[idx]]
                format_data[tag]['gt_labels'] = [gt_labels[idx]]
            else:
                format_data[tag]['img'].append(imgs[idx])
                format_data[tag]['img_metas'].append(img_metas[idx])
                format_data[tag]['gt_bboxes'].append(gt_bboxes[idx])
                format_data[tag]['gt_labels'].append(gt_labels[idx])
        for key in format_data.keys():
            format_data[key]['img'] = torch.stack(format_data[key]['img'], dim=0)
            # print(f"{key}: {format_data[key]['img'].shape}")

        losses = dict()
        # supervised forward 有监督训练
        sup_losses = self.student.forward_train(**format_data['sup'])
        for key, val in sup_losses.items():
            if key[:4] == 'loss':
                if isinstance(val, list):
                    losses[f"{key}_sup"] = [self.sup_weight * x for x in val]
                else:
                    losses[f"{key}_sup"] = self.sup_weight * val
            else:
                losses[key] = val
        if self.iter_count > self.burn_in_steps:
            # Train Logic
            # unsupervised forward 无监督训练
            unsup_weight = self.unsup_weight
            if self.weight_suppress == 'exp':
                target = self.burn_in_steps + 2000
                if self.iter_count <= target:
                    scale = np.exp((self.iter_count - target) / 1000)
                    unsup_weight *= scale
            elif self.weight_suppress == 'step':
                target = self.burn_in_steps * 2
                if self.iter_count <= target:
                    unsup_weight *= 0.25
            elif self.weight_suppress == 'linear':
                target = self.burn_in_steps * 2
                if self.iter_count <= target:
                    unsup_weight *= (self.iter_count - self.burn_in_steps) / self.burn_in_steps

            with torch.no_grad():
                # get teacher data
                teacher_logits = self.teacher.forward_train(
                    get_data=True, **format_data['unsup_weak'])

            # get student data
            student_logits = self.student.forward_train(get_data=True, **format_data['unsup_strong'])
            unsup_losses = self.semi_loss(teacher_logits, student_logits, img_metas=format_data['unsup_weak'], alone_angle=True)
            # T：元组(列表1，列表2，列表3，列表4)
            # 列表1[tensor[2,15,128,128],tensor[2,15,64,64],tensor[2,15,32,32],tensor[2,15,16,16],tensor[2,15,8,8]]
            # 列表2[tensor[2,4,128,128], tensor[2,4,64,64], tensor[2,4,32,32], tensor[2,4,16,16], tensor[2,4,8,8]]
            # 列表3[tensor[2,3,128,128], tensor[2,3,64,64], tensor[2,3,32,32], tensor[2,3,16,16], tensor[2,3,8,8]]
            # 列表4[tensor[2,1,128,128], tensor[2,1,64,64], tensor[2,1,32,32], tensor[2,1,16,16], tensor[2,1,8,8]]
            for key, val in self.logit_specific_weights.items():
                if key in unsup_losses.keys():
                    unsup_losses[key] *= val
            for key, val in unsup_losses.items():
                if key[:4] == 'loss':
                    losses[f"{key}_unsup"] = unsup_weight * val
                else:
                    losses[key] = val
            
            # losses["loss_attention_unsup"] = self.att_loss(student_logits, teacher_logits) 
         
        self.iter_count += 1

        return losses