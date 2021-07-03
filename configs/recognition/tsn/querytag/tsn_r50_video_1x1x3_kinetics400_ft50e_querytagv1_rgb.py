_base_ = [
    '../../../_base_/models/tsn_r50.py', '../../../_base_/schedules/sgd_100e.py',
    '../../../_base_/default_runtime.py'
]

# model settings
num_classes = 3750
model = dict(
    backbone=dict(pretrained='torchvision://resnet50', depth=50),
    cls_head=dict(
        num_classes=num_classes,
        multi_class=True,
        loss_cls=dict(type='BCELossWithLogits', loss_weight=333.)))

# dataset settings
dataset_type = 'VideoDataset'
data_root = 'data/querytag/videos'
data_root_val = 'data/querytag/videos'
ann_file_train = 'data/querytag/querytag_train_list.txt'
ann_file_val = 'data/querytag/querytag_val_list.txt'
ann_file_test = 'data/querytag/querytag_test_list.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=3),
    dict(type='DecordDecode'),
    dict(type='RandomResizedCrop'),
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
        num_clips=3,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
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
        num_clips=25,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(256, -1)),
    dict(type='TenCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=32,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=8),
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
    interval=2,
    metrics=['mmit_mean_average_precision', 'top_k_precision', 'top_k_recall'], # mmit: sample-based. mAP
    metric_options=dict(top_k_precision=dict(topk=(1, 3, 5, 10)), top_k_recall=dict(topk=(3, 5, 10))),
    save_best='mmit_mean_average_precision')

# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)  # 0.01 lr is used for 8 gpus, train from scratch
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

# learning policy
lr_config = dict(policy='step', step=[40, 80])
total_epochs = 100
checkpoint_config = dict(interval=2)
log_config = dict(  # 注册日志钩子的设置
    interval=20,  # 打印日志间隔
    hooks=[  # 训练期间执行的钩子
        dict(type='TextLoggerHook'),  # 记录训练过程信息的日志
        dict(type='TensorboardLoggerHook'),  # 同时支持 Tensorboard 日志
    ])

# use the pre-trained model for the whole TSN network
load_from = 'https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_video_320p_1x1x3_100e_kinetics400_rgb_20201014-5ae1ee79.pth'  # model path can be found in model zoo

# runtime settings
work_dir = './work_dirs/tsn_r50_video_1x1x3_kinetics400_ft50e_querytagv1_rgb/'
