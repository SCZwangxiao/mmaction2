_base_ = [
    '../../_base_/default_runtime.py'
]

# model settings
num_classes = 5911
model = dict(
    type='RecognizerTagRel',
    backbone=dict(
        type='ResNet',
        pretrained='torchvision://resnet50',
        depth=50,
        norm_eval=False),
    cls_head=dict(
        type='NextVLADHead',
        online_model=True,
        num_classes=num_classes,
        multi_class=True,
        loss_cls=dict(
            type='AsymmetricFocalLoss', 
            gamma_neg=1,
            gamma_pos=0,
            disable_torch_grad_focal_loss=True),
        in_channels=2048,
        cluster_size=128,
        expansion=1,
        groups=16,
        hidden_size=2048,
        gating_reduction=8,
        dropout_ratio=0.4),
    # model training and testing settings
    train_cfg=dict(aux_info=['vertical']),
    test_cfg=dict(average_clips=None))

# dataset settings
dataset_type = 'KwaiFeatureDataset'
data_root = '/home/wangxiao13/annotation/data/relevance/frames_feat'
ann_file_train = '/home/wangxiao13/annotation/data/relevance/tagrel_preci_train.txt'
ann_file_val = '/home/wangxiao13/annotation/data/relevance/tagrel_preci_val.txt'
ann_file_test = '/home/wangxiao13/annotation/data/relevance/tagrel_preci_test.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='Collect', keys=['imgs', 'vertical', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'vertical'])
]
val_pipeline = [
    dict(type='Collect', keys=['imgs', 'vertical', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'vertical'])
]
test_pipeline = [
    dict(type='Collect', keys=['imgs', 'vertical', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'vertical'])
]
data = dict(
    videos_per_gpu=128,
    workers_per_gpu=6, # 32-2-TSN 128-4
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
    interval=500, 
    by_epoch=False,
    metrics=['mmit_mean_average_precision', 'mean_average_precision',
            'top_k_precision', 'top_k_recall'],  # mmit: sample-based. mAP
    metric_options=dict(top_k_precision=dict(topk=(1, 3, 5)),
                        top_k_recall=dict(topk=(5, 10))),
    save_best='mmit_mean_average_precision')

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0002, # 0.01 is used for 8 gpus 32 videos/gpu
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))

# learning policy
lr_config = dict(
    policy='step', 
    step=[1, 2, 3, 4, 5],
    gamma=0.5,
    by_epoch=True)
total_epochs = 8

# runtime settings
checkpoint_config = dict(interval=500, by_epoch=False)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])

# load_from = 'https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_256p_1x1x8_100e_kinetics400_rgb/tsn_r50_256p_1x1x8_100e_kinetics400_rgb_20200817-883baf16.pth'  # noqa: E501