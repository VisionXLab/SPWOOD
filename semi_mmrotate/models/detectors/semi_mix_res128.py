# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.functional import grid_sample

from mmrotate.models.builder import ROTATED_DETECTORS
from mmrotate.models.detectors.single_stage import RotatedSingleStageDetector
from torchvision import transforms

from semi_mmrotate.models.third_parties.ted.ted import TED


class SimpleFPNDecoder(nn.Module):
    def __init__(self,
                 in_channels_list=[256, 256, 256, 256, 256], # 对应 FPN 输出 P2, P3, P4, P5, P6 的通道数
                 out_channels=3, # RGB图像为3通道
                 norm_cfg=None,  # 可选的BN/GN配置
                 act_cfg=dict(type='ReLU'), # 激活函数
                 upsample_mode='bilinear'): # 上采样模式
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.upsample_mode = upsample_mode

        # 2. 从 P5 -> P4 -> P3 -> P2 逐步上采样和融合
        # 注意：这里假设输入feat列表的顺序是 [P2, P3, P4, P5, P6]
        # 如果是 [P6, P5, P4, P3, P2] 则需要调整索引

        # 从 P5 融合到 P4
        # self.lateral_conv_p4 = nn.Conv2d(in_channels_list[-2], in_channels_list[-2], 1) # FPN内部已经融合，这里可能不需要额外的横向连接

        # P4 的解码处理层
        self.decoder_block_p4 = nn.Sequential(
            nn.Conv2d(in_channels_list[-2], in_channels_list[-3], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels_list[-3], in_channels_list[-3], 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # P3 的解码处理层
        self.decoder_block_p3 = nn.Sequential(
            nn.Conv2d(in_channels_list[-3], in_channels_list[-4], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels_list[-4], in_channels_list[-4], 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # P2 的解码处理层
        self.decoder_block_p2 = nn.Sequential(
            nn.Conv2d(in_channels_list[-4], in_channels_list[-4], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels_list[-4], in_channels_list[-4], 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 3. 最终输出层，将 P2 级别特征转换为图像
        # 假设最终输出图像分辨率与 P2 相同，或在此基础上再上采样一层
        self.output_conv = nn.Conv2d(in_channels_list[-4], out_channels, 1) # 1x1 卷积调整通道数到3

    def forward(self, feats, w, h):
        # feats 应该是从 P2 到 P6 的列表： [p2_feat, p3_feat, p4_feat, p5_feat, p6_feat]
        # 你的 feat 列表中顺序是 [P2, P3, P4, P5, P6] 对应索引 0,1,2,3,4

        # 从最深层特征开始解码
        p6 = feats[4] # 8x8
        p5 = feats[3] # 16x16
        p4 = feats[2] # 32x32
        p3 = feats[1] # 64x64
        p2 = feats[0] # 128x128

        # 从 P6 开始上采样，然后融合 P5
        # 这里的具体融合方式可以有多种，例如直接相加，或通过卷积后再相加
        # 这是一个简化的示例，假设 FPN 已经提供了融合好的特征，我们主要做上采样
        
        # P6 上采样到 P5 尺寸
        x = F.interpolate(p6, size=p5.shape[2:], mode=self.upsample_mode, align_corners=False)
        x = x + p5 # 特征融合 (这里只是简单相加，也可以通过conv操作)
        x = self.decoder_block_p4(x) # 经过 P4 解码块

        # x (现在是 P4 尺寸) 上采样到 P3 尺寸
        x = F.interpolate(x, size=p3.shape[2:], mode=self.upsample_mode, align_corners=False)
        x = x + p3 # 特征融合
        x = self.decoder_block_p3(x) # 经过 P3 解码块

        # x (现在是 P3 尺寸) 上采样到 P2 尺寸
        x = F.interpolate(x, size=p2.shape[2:], mode=self.upsample_mode, align_corners=False)
        x = x + p2 # 特征融合
        x = self.decoder_block_p2(x) # 经过 P2 解码块

        # 最终输出层：将通道数调整为3，并进行最后的上采样（如果 P2 还不够原始图像分辨率）
        # 你的原图和增强图可能是 512x512 或 1024x1024。
        # 如果原图是 512x512，P2 (128x128) 还需要再上采样 4 倍
        # 如果原图是 1024x1024，P2 (128x128) 还需要再上采样 8 倍

        # 假设原图尺寸为 (H, W)
        # 这里的 size 参数需要根据你的实际输入图像大小来设定
        # 你可以从 batch_inputs_all 中获取原始图像的 H 和 W
        original_h, original_w = h, w # 假设你的原始图像是 512x512

        reconstructed_image = self.output_conv(x)
        # 如果 P2 仍然不是原始分辨率，需要最终的上采样
        reconstructed_image = F.interpolate(reconstructed_image, 
                                        size=(original_h, original_w), 
                                        mode=self.upsample_mode, 
                                        align_corners=False)
        
        # 将输出限制在 [0, 1] 范围或 [-1, 1] 范围，取决于你原图的归一化方式
        # 通常可以是一个 sigmoid 激活函数，或者 tanh
        # 例如：return torch.sigmoid(reconstructed_image) # 如果希望输出在0-1之间
        
        return reconstructed_image


def plot_one_rotated_box(img,
                         obb,
                         color=[0.0, 0.0, 128],
                         label=None,
                         line_thickness=None):
    width, height, theta = obb[2], obb[3], obb[4] / np.pi * 180
    if theta < 0:
        width, height, theta = height, width, theta + 90
    rect = [(obb[0], obb[1]), (width, height), theta]
    poly = np.intp(np.round(
        cv2.boxPoints(rect)))  # [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    cv2.drawContours(
        image=img, contours=[poly], contourIdx=-1, color=color, thickness=2)
    c1 = (int(obb[0]), int(obb[1]))
    if label:
        tl = 2
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        textcolor = [0, 0, 0] if max(color) > 192 else [255, 255, 255]
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, textcolor, thickness=tf, lineType=cv2.LINE_AA)

@ROTATED_DETECTORS.register_module()
class SemiMixRes128(RotatedSingleStageDetector):
    """Implementation of `H2RBox-v2 <https://arxiv.org/abs/2304.04403>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 rotate_range = (0.25, 0.75),
                 scale_range = (0.5, 0.9),
                 ss_prob = [0.6, 0.15, 0.25],
                 copy_paste_start_iter = 60000,
                 num_copies = 10,
                 debug = False,
                 train_cfg = None,
                 test_cfg = None,
                 pretrained=None,
                 init_cfg = None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.rotate_range = rotate_range
        self.scale_range = scale_range
        self.ss_prob = ss_prob
        self.copy_paste_start_iter = copy_paste_start_iter
        self.num_copies = num_copies
        self.debug = debug
        self.copy_paste_cache = None

        self.ted_model = TED()
        for param in self.ted_model.parameters():
            param.requires_grad = False
        self.ted_model.load_state_dict(torch.load('semi_mmrotate/models/third_parties/ted/ted.pth'))
        self.ted_model.eval()

        
        self.FPNdecoder = SimpleFPNDecoder(
            in_channels_list=[256, 256, 256, 256, 256], # 根据你的FPN输出通道数设置
            out_channels=3 # RGB图像
        )

    # def set_epoch(self, epoch):
    #     self.epoch = epoch
    #     self.bbox_head.epoch = epoch

    def rotate_crop(
            self,
            batch_inputs,
            rot = 0.,
            size = (768, 768),
            batch_gt_instances = None,
            padding = 'reflection'):
        """

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            rot (float): Angle of view rotation. Defaults to 0.
            size (tuple[int]): Crop size from image center.
                Defaults to (768, 768).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  
                padding (str): Padding method of image black edge.
                Defaults to 'reflection'.

        Returns:
            Processed batch_inputs (Tensor) and batch_gt_instances
            (list[:obj:`InstanceData`])
        """
        device = batch_inputs.device
        n, c, h, w = batch_inputs.shape
        size_h, size_w = size
        crop_h = (h - size_h) // 2 
        crop_w = (w - size_w) // 2 
        if rot != 0:
            cosa, sina = math.cos(rot), math.sin(rot)
            tf = batch_inputs.new_tensor([[cosa, -sina], [sina, cosa]],
                                         dtype=torch.float)
            x_range = torch.linspace(-1, 1, w, device=device) 
            y_range = torch.linspace(-1, 1, h, device=device)
            y, x = torch.meshgrid(y_range, x_range)
            grid = torch.stack([x, y], -1).expand([n, -1, -1, -1]) 
            grid = grid.reshape(-1, 2).matmul(tf).view(n, h, w, 2) 
            # rotate
            batch_inputs = grid_sample(
                batch_inputs, grid, 'bilinear', padding, align_corners=True)
            if batch_gt_instances is not None:
                for i, gt_instances in enumerate(batch_gt_instances):
                    gt_bboxes = gt_instances
                    xy, wh, a = gt_bboxes[..., :2], gt_bboxes[  
                        ..., 2:4], gt_bboxes[..., [4]]
                    ctr = tf.new_tensor([[w / 2, h / 2]])
                    xy = (xy - ctr).matmul(tf.T) + ctr
                    a = a + rot
                    rot_gt_bboxes = torch.cat([xy, wh, a], dim=-1)
                    batch_gt_instances[i] = rot_gt_bboxes
        batch_inputs = batch_inputs[..., crop_h:crop_h + size_h,
                                    crop_w:crop_w + size_w]
        if batch_gt_instances is None:
            return batch_inputs
        else:  # rot == 0
            for i, gt_instances in enumerate(batch_gt_instances):
                gt_bboxes = gt_instances
                xy, wh, a = gt_bboxes[..., :2], gt_bboxes[...,
                                                          2:4], gt_bboxes[...,
                                                                          [4]]
                xy = xy - xy.new_tensor([[crop_w, crop_h]])
                crop_gt_bboxes = torch.cat([xy, wh, a], dim=-1)
                batch_gt_instances[i] = crop_gt_bboxes

            return batch_inputs, batch_gt_instances
    

    def forward_train(self, 
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      get_data=False,
                      rsst_flag=False,
                      gt_bboxes_ignore=None,
                      need_res=False):
        """Calculate losses from a batch of inputs and data samples.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            img_metas (list[dict]): The batch image metadata.
            gt_bboxes (list[Tensor]): Ground truth bounding boxes for each image.
            gt_labels (list[Tensor]): Class labels for each ground truth box.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes that are
                ignored during training. Defaults to None.

        Returns:
            dict: A dictionary of loss components.
        """
        H, W = img.shape[2:4]
        batch_gt_instances = []
        
        self.bbox_head.iter_count = self.iter_count
        
        # Convert gt_bboxes and gt_labels into structured instance format
        for i in range(len(gt_bboxes)):
            instance = {'bboxes': gt_bboxes[i], 'labels': gt_labels[i]}
            batch_gt_instances.append(instance)

        offset = 1
        for i, gt_instances in enumerate(batch_gt_instances):
            blen = len(gt_instances['bboxes'])
            bids = gt_instances['labels'].new_zeros(blen, 4)
            bids[:, 0] = i
            bids[:, 3] = torch.arange(0, blen, 1) + offset
            gt_instances['bids'] = bids
            offset += blen

        sel_p = torch.rand(1)
        if sel_p < self.ss_prob[0]:
            # Generate rotated images and gts
            rot = math.pi * (
                torch.rand(1).item() *
                (self.rotate_range[1] - self.rotate_range[0]) + self.rotate_range[0])
            for meta in img_metas:
                meta['ss'] = ('rot', rot)
            img_aug = transforms.functional.rotate(img, -rot / math.pi * 180)
            cosa, sina = math.cos(rot), math.sin(rot)
            tf = img.new_tensor([[cosa, -sina], [sina, cosa]], dtype=torch.float)
            ctr = tf.new_tensor([[img.shape[-1] / 2, img.shape[-2] / 2]])
            batch_gt_aug = copy.deepcopy(batch_gt_instances)
            # img_aug, batch_gt_aug = self.rotate_crop(img, rot, [H, W], batch_gt_instances, 'reflection')
            for gt_instances in batch_gt_aug:
                gt_instances['bboxes'][:, :2] = (gt_instances['bboxes'][..., :2] - ctr).matmul(tf.T) + ctr
                gt_instances['bboxes'][:, 4] = gt_instances['bboxes'][:, 4] + rot
                gt_instances['bids'][:, 0] += len(batch_gt_instances)
                gt_instances['bids'][:, 2] = 1
        elif sel_p < self.ss_prob[0] + self.ss_prob[1]:
            # Generate flipped images and gts
            for meta in img_metas:
                meta['ss'] = ('flp', 0)
            img_aug = transforms.functional.vflip(img)
            batch_gt_aug = copy.deepcopy(batch_gt_instances)
            for gt_instances in batch_gt_aug:
                gt_instances['bboxes'][:, 1] = img.shape[-2] - gt_instances['bboxes'][:, 1]
                gt_instances['bboxes'][:, 4] = -gt_instances['bboxes'][:, 4]
                gt_instances['bids'][:, 0] += len(batch_gt_instances)
                gt_instances['bids'][:, 2] = 1
        else:
            # Generate scaled images and gts
            sca = (torch.rand(1).item() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0])
            for meta in img_metas:
                meta['ss'] = ('sca', sca)
            img_aug = transforms.functional.resized_crop(img, 0, 0, int(H / sca), int(W / sca), [H, W])
            batch_gt_aug = copy.deepcopy(batch_gt_instances)
            for gt_instances in batch_gt_aug:
                gt_instances['bboxes'][:, :4] *= sca
                gt_instances['bids'][:, 0] += len(batch_gt_instances)
                gt_instances['bids'][:, 2] = 1
                
        img_all = torch.cat((img, img_aug))
        self.bbox_head.images = img_all
        # Edge
        if self.iter_count >= self.bbox_head.edge_loss_start_iter:
            with torch.no_grad():
                mean = img_all.new_tensor([123.675, 116.28, 103.53])[..., None, None]
                std = img_all.new_tensor([58.395, 57.12, 57.375])[..., None, None]
                batch_edges = self.ted_model(img_all * std + mean)
                self.bbox_head.edges = batch_edges[3].clamp(0)
                # cv2.imwrite('E.png', batch_edges[0].cpu().numpy() * 255)
        
        # # 为了rsst针对单张图片的loss新加  可以舍弃了,因为数据增强的效果肯定比单张图片要好
        # if rsst_flag:
        #     batch_inputs_all = img
        #     feat = self.extract_feat(batch_inputs_all)
        #     cls_score, bbox_pred, angle_pred, centerness = self.bbox_head.forward(feat, get_data)
        #     return (cls_score, bbox_pred, angle_pred, centerness)

        batch_inputs_all = torch.cat((img, img_aug))
        batch_data_samples_all = []
        for gt_instances, img_metas in zip(batch_gt_instances + batch_gt_aug, img_metas + img_metas):
            data_sample = {'metainfo': img_metas, 'gt_instances': gt_instances}
            batch_data_samples_all.append(data_sample)
        # 有监督,batch_inputs_all:torch.Size([4, 3, 1024, 1024])
        feat = self.extract_feat(batch_inputs_all)  #feat [4, 256, 128, 128][4, 256, 64, 64][4, 256, 32, 32][4, 256, 16, 16][4, 256, 8, 8]
        # cls_score:list[torch.Size([4, 15, 128, 128]), .......]
        cls_score, bbox_pred, angle_pred, centerness = self.bbox_head.forward(feat, get_data)
        
        # batch_gt_instances = [data_sample['gt_instances'] for data_sample in batch_data_samples_all]
        batch_gt_instances = [copy.deepcopy(data_sample['gt_instances']) for data_sample in batch_data_samples_all]
        batch_img_metas = [data_sample['metainfo'] for data_sample in batch_data_samples_all]
          
        if rsst_flag:  # label无监督需要返回GT
            return (cls_score, bbox_pred, angle_pred, centerness, batch_gt_instances)

        if get_data:   # unlabel无监督时候使用的是使用img得到的feat得到的预测结果.和GT没有关系
            return (cls_score, bbox_pred, angle_pred, centerness)
        
        # 这里使用所有图片的FPN第一层的结果
        results_list = self.bbox_head.get_bboxes((cls_score[0],), 
                                                 (bbox_pred[0],), 
                                                 (angle_pred[0],),
                                                 (centerness[0],), 
                                                 batch_img_metas, 
                                                 batch_gt_instances=batch_gt_instances)
        converted_results_list = []
        for det_bboxes, det_labels in results_list:
            bboxes = det_bboxes[:, :5]
            scores = det_bboxes[:, 5]
            result_dict = {
                'bboxes': bboxes,
                'scores': scores,  
                'labels': det_labels 
            }
            converted_results_list.append(result_dict)
        
        # Update point annotations with predicted rbox 水平框的没有变化
        for data_sample, results in zip(batch_gt_instances, converted_results_list):
            mask = data_sample['bids'][:, 1] == 0
            data_sample['bboxes'][mask] = results['bboxes'][mask]
            data_sample['labels'][mask] = results['labels'][mask]

        losses = self.bbox_head.loss(cls_score,
                                     bbox_pred,
                                     angle_pred,
                                     centerness,
                                     batch_gt_instances,
                                     batch_img_metas)


        if need_res and self.iter_count <= 12800:  # 在burn in阶段，对于sparse label进行语义学习
            batch_inputs_all_mask, mask = self.mask_image_function(batch_inputs_all, batch_gt_instances, img.shape[2], img.shape[3])
            feat = self.extract_feat(batch_inputs_all_mask) 
            reconstructed_batch = self.FPNdecoder(feat, img.shape[2], img.shape[3]) 
            # 例如，假设你有一个二值mask，1表示被挖空区域，0表示未挖空区域
            inverted_mask = 1 - mask

            loss_reconstruction = (reconstructed_batch - batch_inputs_all).abs() * inverted_mask
            loss_reconstruction = loss_reconstruction.sum() / (inverted_mask.sum() + 1e-5) # 仅在mask区域求平均          
            losses['loss_reconstruction'] = loss_reconstruction

        # self.debug = True
        if self.debug:
            for i in range(len(batch_inputs_all)):
                img = batch_inputs_all[i]
                if self.bbox_head.vis[i]:
                    vor, wat = self.bbox_head.vis[i]
                    img[0, wat != wat.max()] += 2
                    img[:, vor != vor.max()] -= 1
                img = img.permute(1, 2, 0).cpu().numpy()
                img = np.ascontiguousarray(img[..., (2, 1, 0)] * 58 + 127)
                bb = batch_data_samples_all[i]['gt_instances']['bboxes']
                ll = batch_data_samples_all[i]['gt_instances']['labels']
                for b, l in zip(bb.cpu().numpy(), ll.cpu().numpy()):
                    b[2:4] = b[2:4].clip(3)
                    plot_one_rotated_box(img, b, (255, 0, 0))
                if i < len(converted_results_list):
                    bb = converted_results_list[i]['bboxes']
                    if hasattr(converted_results_list[i], 'informs'):
                        for b, l in zip(bb.cpu().numpy(), converted_results_list[i].infoms.cpu().numpy()):
                            plot_one_rotated_box(img, b, (0, 255, 0), label=f'{l}')
                    else:
                        for b in bb.cpu().numpy():
                            plot_one_rotated_box(img, b, (0, 255, 0))
                # img_id = batch_data_samples_all[i]['metainfo']['filename']
                img_id = i
                img = np.clip(img, 0, 255).astype(np.uint8)
                cv2.imwrite(f'./show/{img_id}.png', img)
        
        return losses


    def mask_image_function(self, batch_inputs_all, batch_gt_instances, w, h):
        device = batch_inputs_all.device
        batch_size = len(batch_gt_instances)
        height, width = h, w

        # 在指定设备上初始化一个全1的批次mask Tensor
        # 形状为 (B, 1, H, W)，满足你的最终需求
        batch_masks = torch.ones((batch_size, 1, height, width), dtype=torch.float32, device=device)

        for idx, data_sample in enumerate(batch_gt_instances):
            single_img_bboxes = data_sample['bboxes'] # 这里假设bboxes已经是Nx4的AABB格式
            # 如果一个图片没有任何标注实例，则跳过，mask保持全1
            if single_img_bboxes.numel() == 0:
                continue # mask 已经初始化为全1，无需修改
            # 假设已经是torch.Tensor
            bboxes_tensor = single_img_bboxes.to(device)

            # 遍历当前图片的所有矩形框
            for bbox_coords in bboxes_tensor:
                x_min = int(max(0, bbox_coords[0]-bbox_coords[2]/2))
                y_min = int(max(0, bbox_coords[1]-bbox_coords[3]/2))
                x_max = int(min(width - 1, bbox_coords[0]+bbox_coords[2]/2)) # 确保不越界
                y_max = int(min(height - 1, bbox_coords[1]+bbox_coords[3]/2)) # 确保不越界
                
                # 将矩形区域设置为0
                # 注意：这里直接修改批次mask中对应图片索引的通道0
                batch_masks[idx, 0, y_min:y_max+1, x_min:x_max+1] = 0

        # 将标注框内的部分设置为0，从而将图片中的对应部分抹去。
        batch_inputs_all_mask = batch_inputs_all * batch_masks
        
        return batch_inputs_all_mask, batch_masks
    

     
     


