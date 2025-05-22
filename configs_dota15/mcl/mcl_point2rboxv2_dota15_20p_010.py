angle_version = 'le90'

import torchvision.transforms as transforms
from copy import deepcopy

# 模型核心组件: 检测器(包括backbone/neck/head/训练测试配置)
detector = dict(
    type='SemiPoint2RBoxV2',
    ss_prob=[0.68, 0.07, 0.25],  # 自监督变换比例
    backbone=dict(
        type='ResNet',  # 模块的类型为ResNet
        depth=50,  # ResNet-50
        num_stages=4,  # 模型的阶段
        out_indices=(1, 2, 3),  # ResNet的输出特征层索引为第1、2、3阶段。
        frozen_stages=1,  # 冻结第1阶段的权重   
        norm_cfg=dict(type='BN', requires_grad=True),  # 定义归一化层的配置,批量归一化,参数需要更新
        norm_eval=True,  # 推理阶段将归一化层设置为评估模式
        style='pytorch',  # ResNet的实现风格为PyTorch风格
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),  # ResNet-50预训练权重来初始化
    neck=dict(
        type='FPN',  # 模块的类型为FPN
        in_channels=[512, 1024, 2048],  # ResNet的不同阶段输出的特征图
        out_channels=128,  # FPN输出的每个特征图的通道数
        start_level=0,  # 从第一个输入特征图(通道数512)开始构建特征金字塔
        add_extra_convs='on_output',  # FPN在输出特征图上添加额外的卷积层
        num_outs=3,  # FPN输出的特征图数量,3张特征图,通道数为128
        relu_before_extra_convs=True),  # 额外卷积层之前应用ReLU激活函数
    bbox_head=dict(
        type='SemiPoint2RBoxV2Head',  # 模块的类型为SemiPoint2RBoxV2Head
        num_classes=16,  # 目标检测任务中的类别数量，不包括背景
        in_channels=128,  # 输入特征图的通道数，从FPN
        feat_channels=128,  # 头部模块内部使用的特征通道数
        strides=[8],
        voronoi_type='standard',  # Voronoi图的类型
        voronoi_thres=dict(  # Voronoi图的阈值设置
            default=[0.994, 0.005],
            override=(([2, 11], [0.999, 0.6]),
                    ([7, 8, 10, 14], [0.95, 0.005]))),
        square_cls=[1, 9, 11],  # 指定哪些类别被认为是“正方形”类别
        edge_loss_cls=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13],  # 指定哪些类别用于边缘损失计算
        post_process={},
        angle_coder=dict(
            type='PSCCoder',
            angle_version='le90',
            dual_freq=False,
            num_step=3,
            thr_mod=0),
        loss_cls=dict(  # 类别损失
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_overlap=dict(  # 高斯重叠损失
            type='GaussianOverlapLoss', loss_weight=10.0, lamb=0),
        loss_voronoi=dict(  # Voronoi图损失
            type='GaussianVoronoiLoss', loss_weight=5.0),
        loss_bbox_edg=dict(  # 边缘损失
            type='EdgeLoss', loss_weight=0.3),
        loss_ss=dict(  # 弱分支损失
            type='mmdet.SmoothL1Loss', loss_weight=1.0, beta=0.1)),
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,  # 限制进入NMS的检测框数量
        min_bbox_size=0,  # 不限制框大小
        score_thr=0.05,  # 分数阈值以减少框数
        nms=dict(iou_thr=0.1),  # 非极大值抑制NMS配置
        max_per_img=2000))  # NMS保留的最大检测框数量

# 定义完整模型(包括检测器/损失函数/训练测试配置)
model = dict(
    type="MCLTeacher",  # 模型的类型为MCLTeacher
    model=detector,
    semi_loss=dict(type='RotatedMCLLoss2', cls_channels=16),  # 定义半监督学习损失函数。协调不同模块
    train_cfg=dict(
        iter_count=0,
        burn_in_steps=12800,
        sup_weight=1.0,
        unsup_weight=1.0,
        weight_suppress="linear",
        logit_specific_weights=dict(),
    ),
    test_cfg=dict(inference_on="teacher"),  # 推理时使用教师模型   
)

find_unused_parameters=True

# 数据预处理
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
common_pipeline = [
    dict(type='Normalize', **img_norm_cfg),  # Normalize
    dict(type='Pad', size_divisor=32),  # 对图像进行填充Padding使其尺寸能够被size_divisor整除
    dict(type='DefaultFormatBundle'),  # 格式化
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],  # 收集数据和元数据
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 'flip',
                    'flip_direction', 'img_norm_cfg', 'tag')
         )
]
strong_pipeline = [  # 无标签强处理
    dict(type='DTToPILImage'),  # 输入图像转换为PIL图像
    dict(type='DTRandomApply', operations=[transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    dict(type='DTRandomGrayscale', p=0.2),
    dict(type='DTRandomApply', operations=[
        dict(type='DTGaussianBlur', rad_range=[0.1, 2.0])
    ]),
    # dict(type='DTRandCrop'),
    dict(type='DTToNumpy'),  # PIL图像转换为NumPy数组
    dict(type="ExtraAttrs", tag="unsup_strong"),  # 数据添加额外的标签
]
weak_pipeline = [  # 无标签弱处理
    # dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type="ExtraAttrs", tag="unsup_weak"),
]
unsup_pipeline = [  # 无标签处理
    dict(type="LoadImageFromFile"),
    # dict(type="LoadAnnotations", with_bbox=True),
    # generate fake labels for data format compatibility
    dict(type="LoadEmptyAnnotations", with_bbox=True),
    dict(type="STMultiBranch", unsup_strong=deepcopy(strong_pipeline), unsup_weak=deepcopy(weak_pipeline),
         common_pipeline=common_pipeline, is_seq=True), 
]
sup_pipeline = [  # 有标签处理
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='ConvertWeakSupervision', #标注数据的比例分布（rbox/hbox/point）
         point_proportion=0.,
         hbox_proportion=1,
         modify_labels=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type="ExtraAttrs", tag="sup_weak"),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 'flip',
                    'flip_direction', 'img_norm_cfg', 'tag')
         )
]
test_pipeline = [  # 测试阶段
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

# 数据集
dataset_type = 'DOTADataset'   
classes = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
           'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
           'basketball-court', 'storage-tank', 'soccer-ball-field',
           'roundabout', 'harbor', 'swimming-pool', 'helicopter',
           'container-crane')
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=5,
    train=dict(
        type="SemiDataset",
        sup=dict(  # 标签
            type=dataset_type,
            ann_file="/mnt/nas2/home/yangxue/lmx/data/semi/train_20p_labeled/annfiles/",
            img_prefix="/mnt/nas2/home/yangxue/lmx/data/semi/train_20p_labeled/images/",
            classes=classes,
            pipeline=sup_pipeline,
        ),
        unsup=dict(
            type=dataset_type,
            ann_file="/mnt/nas2/home/yangxue/lmx/data/semi/train_20p_unlabeled/annfiles/",
            img_prefix="/mnt/nas2/home/yangxue/lmx/data/semi/train_20p_unlabeled/images/",
            classes=classes,
            pipeline=unsup_pipeline,
            filter_empty_gt=False,
        ),
    ),
    val=dict(  # val和test一致
        type=dataset_type,
        img_prefix="/mnt/nas2/home/yangxue/lmx/data/semi/val/images/",
        ann_file="/mnt/nas2/home/yangxue/lmx/data/semi/val/annfiles/",
        classes=classes,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        img_prefix="/mnt/nas2/home/yangxue/lmx/data/semi/val/images/",
        ann_file="/mnt/nas2/home/yangxue/lmx/data/semi/val/annfiles/",
        classes=classes,
        pipeline=test_pipeline,
    ),
    sampler=dict(
        train=dict(
            type="MultiSourceSampler",
            sample_ratio=[2, 1],
            seed=42
        )
    ),
)

custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="WeightSummary"),
    dict(type="MeanTeacher", momentum=0.9996, interval=1, start_steps=3200),
]

# evaluation
evaluation = dict(type="SubModulesDistEvalHook", interval=3200, metric='mAP',
                  save_best='mAP')

# optimizer
# optimizer = dict(type='SGD', lr=0.00025, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

optimizer=dict(
    type='AdamW',
    lr=0.00005,
    betas=(0.9, 0.999),
    weight_decay=0.05)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=120000)
# 120k iters is enough for DOTA
runner = dict(type="IterBasedRunner", max_iters=120000)
checkpoint_config = dict(by_epoch=False, interval=3200, max_keep_ckpts=5)

# Default: disable fp16 training
# fp16 = dict(loss_scale="dynamic")

log_config = dict(
    _delete_=True,
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(
        #     type="WandbLoggerHook",
        #     init_kwargs=dict(
        #         project="rotated_DenseTeacher_10percent",
        #         name="default_bce4cls",
        #     ),
        #     by_epoch=False,
        # ),
    ],
)

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]   # mode, iters

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# custom_imports = dict(
#     imports=['semi_mmrotate'],
#     allow_failed_imports=False)