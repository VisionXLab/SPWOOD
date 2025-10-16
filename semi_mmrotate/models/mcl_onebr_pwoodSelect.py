import torch
import numpy as np
from .rotated_semi_detector_onebr import RotatedSemiDetectorOnebr
from mmrotate.models.builder import ROTATED_DETECTORS
from mmrotate.models import build_detector

def check_model_nan(model, flag):
    for name, param in model.named_parameters():
        if param.requires_grad:  # 只检查需要梯度的参数
            if torch.isnan(param).any():
                if flag==0:  # 学生
                    print(f"student [NaN detected] Parameter '{name}' contains NaN values!")
                else:
                    print(f"teacher [NaN detected] Parameter '{name}' contains NaN values!")
                return True
    print("No NaN found in model parameters.")
    return False

def check_grad_nan(model, flag):
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if torch.isnan(param.grad).any():
                if flag == 0:  # 学生
                    print(f"student [NaN detected] Gradient of parameter '{name}' contains NaN values!")
                else:
                    print(f"teacher [NaN detected] Gradient of parameter '{name}' contains NaN values!")
                return True
    print("No NaN found in model gradients.")
    return False

def check_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print(f'Gradient norm: {total_norm:.4f}')
    return total_norm



@ROTATED_DETECTORS.register_module()
class MCLTeacherOneBrPwoodSelect(RotatedSemiDetectorOnebr):
    def __init__(self, model: dict, semi_loss_unsup, train_cfg=None, test_cfg=None):
        super(MCLTeacherOneBrPwoodSelect, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),  # 初始化teacher/student
            semi_loss_unsup,
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
            self.region_ratio = train_cfg.get("region_ratio")

    def forward_train(self, imgs, img_metas, **kwargs):
        super(MCLTeacherOneBrPwoodSelect, self).forward_train(imgs, img_metas, **kwargs)  # ???
        torch.autograd.set_detect_anomaly(True)
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

        # check_grad_nan(self.student, 0)
        # check_model_nan(self.student, 0)

        # check_grad_nan(self.teacher, 1)
        # check_model_nan(self.teacher, 1)

        # if check_grad_norm(self.teacher) > 5:
        #     print(f'Gradient norm: {check_grad_norm(self.teacher):.4f}')
        # if check_grad_norm(self.student) > 5:
        #     print(f'Gradient norm: {check_grad_norm(self.student):.4f}')

        losses = dict()
        # supervised forward 有监督训练   format_data['sup']的键有 dict_keys(['img', 'img_metas', 'gt_bboxes', 'gt_labels']
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

            
            # unsup部分: 使用rsst的label loss(融合pwood选点策略)和pwood的unlabel loss
            with torch.no_grad():
                # get teacher data
                teacher_logits_unlabled = self.teacher.forward_train(get_data=True, **format_data['unsup_weak_unlabeled'])
                # get student data  [2,5,152,152] [2, 15, 76, 76] [2, 15, 38, 38] [2, 15, 19, 19] 
            student_logits_unlabeled = self.student.forward_train(get_data=True,  **format_data['unsup_strong_unlabeled'])
            # NOTE unlabel和label得到的都是两张图片,例2,15,128,128
            
            
            # 这里使用PWOOD的loss，也就是单层GMM来筛选伪标签 
            unsup_losses_unlabeled = self.semi_loss_unsup(teacher_logits_unlabled, student_logits_unlabeled, img_metas=format_data['unsup_weak_unlabeled'], alone_angle=True)

            for key, val in self.logit_specific_weights.items():
                if key in unsup_losses_unlabeled.keys():
                    unsup_losses_unlabeled[key] *= val
            for key, val in unsup_losses_unlabeled.items():
                # if key[:4] == 'loss':
                if 'loss' in key:
                    # losses[f"{key}_unsup"] = unsup_weight * val
                    losses[f"{key}_unsup_unlabeled"] = val
                else:
                    losses[key] = val

        self.iter_count += 1

        return losses