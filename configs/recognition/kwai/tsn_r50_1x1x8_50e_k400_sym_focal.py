_base_ = [
    '../../_base_/default_runtime.py'
]

# model settings
num_classes = 56769
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='Indentity'),
    cls_head=dict(
        type='VanillaHead',
        num_classes=num_classes,
        multi_class=True,
        loss_cls=dict(
            type='AsymmetricFocalLoss', 
            gamma_neg=2,
            gamma_pos=1,
            disable_torch_grad_focal_loss=True),
        in_channels=2048,
        dropout_ratio=0.4),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips=None))

# dataset settings
dataset_type = 'KwaiFeatureDataset'
data_root = '/home/wangxiao13/annotation/data/relevance/frames_feat'
ann_file_train = '/home/wangxiao13/annotation/data/relevance/tagrel_train.txt'
ann_file_val = '/home/wangxiao13/annotation/data/relevance/tagrel_val.txt'
ann_file_test = '/home/wangxiao13/annotation/data/relevance/tagrel_test.txt'

train_pipeline = [
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=32,
    workers_per_gpu=4, # 32-2-TSN
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        split='train',
        pipeline=train_pipeline,
        total_clips=8,
        num_classes=num_classes),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root,
        split='val',
        pipeline=val_pipeline,
        total_clips=8,
        num_classes=num_classes),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root,
        split='test',
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
    type='Adam',
    lr=0.01, # 0.01 is used for 8 gpus 32 videos/gpu
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

# learning policy
lr_config = dict(policy='Fixed')
total_epochs = 50

# runtime settings
checkpoint_config = dict(interval=5)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])

# work_dir = './work_dirs/tsn_r50_1x1x8_50e_k400_sym_focal/'