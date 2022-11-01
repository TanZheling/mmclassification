_base_ = [
    '../configs/_base_/models/vit-base-p16.py',
    '../configs/_base_/datasets/imagenetc.py',
    '../configs/_base_/schedules/imagenet_bs4096_AdamW.py',
    '../configs/_base_/default_runtime.py'
]


model = dict(
    train_cfg=dict(
        augments=dict(type='BatchMixup', alpha=0.2, num_classes=1000,
                      prob=1.)))
'''data_prefix = '/data/dataset/imagenet'
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix=data_prefix+'/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix=data_prefix+'/val',
        ann_file=None,
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix=data_prefix+'/val',
        ann_file=None,
        pipeline=test_pipeline))'''