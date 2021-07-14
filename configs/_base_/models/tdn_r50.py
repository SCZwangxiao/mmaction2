# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNetTDN',
        pretrained='torchvision://resnet50',
        depth=50,
        norm_eval=False),
    cls_head=dict(
        type='TDNHead',
        num_classes=400,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.01),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips=None))
