_base_ = [
    '../_base_/default_runtime.py'
]

# model settings
num_classes = 3527
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet',
        pretrained='torchvision://resnet50', 
        depth=50,
        frozen_stages=4,
        norm_eval=False),
    cls_head=dict(
        type='TSNHead',
        in_channels=2048,
        num_classes=num_classes,
        multi_class=True,
        spatial_type='avg',
        consensus=dict(
            type='NeXtVLAD', 
            feature_size=2048,
            cluster_size=128,
            expansion=2,
            groups=8,
            hidden_size=2048,
            gating_reduction=8,
            dropout=0.4),
        loss_cls=dict(type='BCELossWithLogits', loss_weight=333.),
        dropout_ratio=0.4,
        init_std=0.01),
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

# dataset settings
dataset_type = 'VideoDataset'
data_root = 'data/querytag/videos'
data_root_val = 'data/querytag/videos'
ann_file_train = 'data/querytag/querytag_v2_train_list.txt'
ann_file_val = 'data/querytag/querytag_v2_val_list.txt'
ann_file_test = 'data/querytag/querytag_v2_test_list.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames', 
        clip_len=1, 
        frame_interval=1, 
        num_clips=16),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(256, -1)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=16,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(256, -1)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=64,
    workers_per_gpu=10,
    test_dataloader=dict(videos_per_gpu=16),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
        multi_class=True,
        num_classes=num_classes),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        multi_class=True,
        num_classes=num_classes),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        multi_class=True,
        num_classes=num_classes))
evaluation = dict(
    interval=1,
    metrics=['mmit_mean_average_precision', 'top_k_precision', 'top_k_recall'], # mmit: sample-based. mAP
    metric_options=dict(top_k_precision=dict(topk=(1, 3, 5, 10)), top_k_recall=dict(topk=(3, 5, 10))),
    save_best='mmit_mean_average_precision')

# optimizer
optimizer = dict(
    type='SGD', 
    lr=0.0025, 
    momentum=0.9, 
    weight_decay=0.0008
)  # 0.001 for 8 gpu batch size 128
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[32, 36])
total_epochs = 40
# checkpoint
checkpoint_config = dict(interval=5)
log_config = dict(  # 注册日志钩子的设置
    interval=20,  # 打印日志间隔
    hooks=[  # 训练期间执行的钩子
        dict(type='TextLoggerHook'),  # 记录训练过程信息的日志
        dict(type='TensorboardLoggerHook'),  # 同时支持 Tensorboard 日志
    ])

load_from = '../arhcived_work_dirs/work_dirs_hetu/tsn_r50_video_1x1x8_kinetics400_ft50e_hetu_rgb/best_mmit_mean_average_precision_epoch_44.pth'
find_unused_parameters = True