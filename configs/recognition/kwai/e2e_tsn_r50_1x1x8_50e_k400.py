_base_ = [
    '../../_base_/default_runtime.py'
]

# model settings
num_classes = 34538
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet',
        pretrained='torchvision://resnet50',
        depth=50,
        norm_eval=False),
    cls_head=dict(
        type='TSNHead',
        num_classes=num_classes,
        multi_class=True,
        loss_cls=dict(type='BCELossWithLogits', loss_weight=333.),
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.4,
        init_std=0.01),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips=None))

# dataset settings
dataset_type = 'KwaiRawframeDataset'
data_root = '/home/wangxiao13/annotation/data/relevance/raw_frames'
data_root_val = '/home/wangxiao13/annotation/data/relevance/raw_frames'
ann_file_train = '/home/wangxiao13/annotation/data/relevance/tagrel_train.txt'
ann_file_val = '/home/wangxiao13/annotation/data/relevance/tagrel_val.txt'
ann_file_test = '/home/wangxiao13/annotation/data/relevance/tagrel_test.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=6,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
        total_clips=8,
        num_classes=num_classes),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        total_clips=8,
        num_classes=num_classes),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        total_clips=8,
        num_classes=num_classes))
evaluation = dict(
    interval=1,
    metrics=['mmit_mean_average_precision', 'mean_average_precision',
            'top_k_precision', 'top_k_recall'],  # mmit: sample-based. mAP
    metric_options=dict(top_k_precision=dict(topk=(1, 3, 5)),
                        top_k_recall=dict(topk=(5, 10))),
    save_best='mmit_mean_average_precision')

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.0025, # 0.01 is used for 8 gpus 32 videos/gpu
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

# learning policy
lr_config = dict(policy='step', step=[20, 40])
total_epochs = 50

# runtime settings
checkpoint_config = dict(interval=5)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])

load_from = 'https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_256p_1x1x8_100e_kinetics400_rgb/tsn_r50_256p_1x1x8_100e_kinetics400_rgb_20200817-883baf16.pth'  # noqa: E501
find_unused_parameters = True

work_dir = './work_dirs/tsn_r50_1x1x8_50e_k400_rgb/'