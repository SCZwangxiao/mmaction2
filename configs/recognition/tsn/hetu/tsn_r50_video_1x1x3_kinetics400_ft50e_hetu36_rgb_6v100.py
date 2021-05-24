_base_ = [
    '../../../_base_/models/tsn_r50.py', '../../../_base_/schedules/sgd_100e.py',
    '../../../_base_/default_runtime.py'
]

# model settings
model = dict(cls_head=dict(num_classes=36))

# dataset settings
dataset_type = 'VideoDataset'
data_root = 'data/hetu/videos'
data_root_val = 'data/hetu/videos'
ann_file_train = 'data/hetu/hetu36_train_list.txt'
ann_file_val = 'data/hetu/hetu36_val_list.txt'
ann_file_test = 'data/hetu/hetu36_test_list.txt'
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
    dict(type='Resize', scale=(-1, 256)),
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
    dict(type='Resize', scale=(-1, 256)),
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
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(type='SGD', lr=0.00375, momentum=0.9, weight_decay=0.0001)  # change from 0.01 to 0.005   # 0.01 lr is used for 8 gpus, fine tune / 2
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

# learning policy
lr_config = dict(policy='step', step=[20, 40]) # change from [40, 80] to [20, 40]
total_epochs = 50 # change from 100 to 50
checkpoint_config = dict(interval=5)

# use the pre-trained model for the whole TSN network
load_from = './work_dirs/tsn_r50_video_320p_1x1x3_100e_hetu_rgb/tsn_r50_video_320p_1x1x3_100e_kinetics400_rgb_20201014-5ae1ee79.pth'  # model path can be found in model zoo

# runtime settings
work_dir = './work_dirs/tsn_r50_video_1x1x3_kinetics400_ft50e_hetu36_rgb_6v100/'