# _base_ = [
#     '../../../_base_/schedules/sgd_100e.py',
#     '../../../_base_/default_runtime.py'
# ]

# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet',
        pretrained='torchvision://resnet50',
        depth=50,
        norm_eval=False),
    cls_head=dict(
        type='TSNHead',
        num_classes=400,
        in_channels=768,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.4,
        init_std=0.01),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips=None))

# dataset settings
dataset_type = 'VideoDataset'
dataset_root = '/home/wangxiao13/dataset/download_video'
data_root = dataset_root + '/video_valid'
data_root_val = dataset_root + '/video_valid'
ann_file_train = dataset_root + '/kwai_video_train_list.txt'
ann_file_val = dataset_root + '/kwai_video_val_list.txt'
ann_file_test = dataset_root + '/kwai_video_test_list.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='DecordDecode', mode='efficient'),
    dict(type='Resize', scale=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='DecordDecode', mode='efficient'),
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
    dict(type='DecordDecode', mode='efficient'),
    dict(type='Resize', scale=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=10,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))

evaluation = dict(
    interval=1,
    metrics=[
        'mmit_mean_average_precision', 'mean_average_precision',
        'top_k_precision', 'top_k_recall'
    ],  # mmit: sample-based. mAP
    metric_options=dict(
        top_k_precision=dict(topk=(1, 3)), top_k_recall=dict(topk=(3, 5))),
    save_best='mmit_mean_average_precision')

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.0075,  # this lr is used for 8 gpus
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

# learning policy
lr_config = dict(policy='step', step=[5, 10])
total_epochs = 15
checkpoint_config = dict(interval=1)
log_config = dict(  # 注册日志钩子的设置
    interval=20,  # 打印日志间隔
    hooks=[  # 训练期间执行的钩子
        dict(type='TextLoggerHook'),  # 记录训练过程信息的日志
        dict(type='TensorboardLoggerHook'),  # 同时支持 Tensorboard 日志
    ])

# runtime settings
checkpoint_config = dict(interval=1)
work_dir = './work_dirs/tsn_swintransformer_feature_extraction_kwai_3x1x8_rgb'
