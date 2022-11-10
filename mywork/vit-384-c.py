_base_ = [
    '../configs/_base_/models/vit-base-p16.py',
    '../configs/_base_/datasets/imagenet_bs64_pil_resize_autoaug.py',
    '../configs/_base_/schedules/imagenet_bs4096_AdamW.py',
    '../configs/_base_/default_runtime.py'
]

model = dict(backbone=dict(img_size=384))

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=384, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(384, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=384),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data_prefix='/home/sjtu/dataset/imagenet'
data = dict(
    train=dict(
        data_prefix=data_prefix+'/train',
        pipeline=train_pipeline),
    val=dict(
        data_prefix=data_prefix+'/val',
        ann_file=None,
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        data_prefix=data_prefix+'/val',
        ann_file=None,
        pipeline=test_pipeline)
)