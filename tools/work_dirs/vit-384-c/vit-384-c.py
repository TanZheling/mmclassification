model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='b',
        img_size=384,
        patch_size=16,
        drop_rate=0.1,
        init_cfg=[
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ]),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1,
            mode='classy_vision')))
policy_imagenet = [[{
    'type': 'Posterize',
    'bits': 4,
    'prob': 0.4
}, {
    'type': 'Rotate',
    'angle': 30.0,
    'prob': 0.6
}],
                   [{
                       'type': 'Solarize',
                       'thr': 113.77777777777777,
                       'prob': 0.6
                   }, {
                       'type': 'AutoContrast',
                       'prob': 0.6
                   }],
                   [{
                       'type': 'Equalize',
                       'prob': 0.8
                   }, {
                       'type': 'Equalize',
                       'prob': 0.6
                   }],
                   [{
                       'type': 'Posterize',
                       'bits': 5,
                       'prob': 0.6
                   }, {
                       'type': 'Posterize',
                       'bits': 5,
                       'prob': 0.6
                   }],
                   [{
                       'type': 'Equalize',
                       'prob': 0.4
                   }, {
                       'type': 'Solarize',
                       'thr': 142.22222222222223,
                       'prob': 0.2
                   }],
                   [{
                       'type': 'Equalize',
                       'prob': 0.4
                   }, {
                       'type': 'Rotate',
                       'angle': 26.666666666666668,
                       'prob': 0.8
                   }],
                   [{
                       'type': 'Solarize',
                       'thr': 170.66666666666666,
                       'prob': 0.6
                   }, {
                       'type': 'Equalize',
                       'prob': 0.6
                   }],
                   [{
                       'type': 'Posterize',
                       'bits': 6,
                       'prob': 0.8
                   }, {
                       'type': 'Equalize',
                       'prob': 1.0
                   }],
                   [{
                       'type': 'Rotate',
                       'angle': 10.0,
                       'prob': 0.2
                   }, {
                       'type': 'Solarize',
                       'thr': 28.444444444444443,
                       'prob': 0.6
                   }],
                   [{
                       'type': 'Equalize',
                       'prob': 0.6
                   }, {
                       'type': 'Posterize',
                       'bits': 5,
                       'prob': 0.4
                   }],
                   [{
                       'type': 'Rotate',
                       'angle': 26.666666666666668,
                       'prob': 0.8
                   }, {
                       'type': 'ColorTransform',
                       'magnitude': 0.0,
                       'prob': 0.4
                   }],
                   [{
                       'type': 'Rotate',
                       'angle': 30.0,
                       'prob': 0.4
                   }, {
                       'type': 'Equalize',
                       'prob': 0.6
                   }],
                   [{
                       'type': 'Equalize',
                       'prob': 0.0
                   }, {
                       'type': 'Equalize',
                       'prob': 0.8
                   }],
                   [{
                       'type': 'Invert',
                       'prob': 0.6
                   }, {
                       'type': 'Equalize',
                       'prob': 1.0
                   }],
                   [{
                       'type': 'ColorTransform',
                       'magnitude': 0.4,
                       'prob': 0.6
                   }, {
                       'type': 'Contrast',
                       'magnitude': 0.8,
                       'prob': 1.0
                   }],
                   [{
                       'type': 'Rotate',
                       'angle': 26.666666666666668,
                       'prob': 0.8
                   }, {
                       'type': 'ColorTransform',
                       'magnitude': 0.2,
                       'prob': 1.0
                   }],
                   [{
                       'type': 'ColorTransform',
                       'magnitude': 0.8,
                       'prob': 0.8
                   }, {
                       'type': 'Solarize',
                       'thr': 56.888888888888886,
                       'prob': 0.8
                   }],
                   [{
                       'type': 'Sharpness',
                       'magnitude': 0.7,
                       'prob': 0.4
                   }, {
                       'type': 'Invert',
                       'prob': 0.6
                   }],
                   [{
                       'type': 'Shear',
                       'magnitude': 0.16666666666666666,
                       'prob': 0.6,
                       'direction': 'horizontal'
                   }, {
                       'type': 'Equalize',
                       'prob': 1.0
                   }],
                   [{
                       'type': 'ColorTransform',
                       'magnitude': 0.0,
                       'prob': 0.4
                   }, {
                       'type': 'Equalize',
                       'prob': 0.6
                   }],
                   [{
                       'type': 'Equalize',
                       'prob': 0.4
                   }, {
                       'type': 'Solarize',
                       'thr': 142.22222222222223,
                       'prob': 0.2
                   }],
                   [{
                       'type': 'Solarize',
                       'thr': 113.77777777777777,
                       'prob': 0.6
                   }, {
                       'type': 'AutoContrast',
                       'prob': 0.6
                   }],
                   [{
                       'type': 'Invert',
                       'prob': 0.6
                   }, {
                       'type': 'Equalize',
                       'prob': 1.0
                   }],
                   [{
                       'type': 'ColorTransform',
                       'magnitude': 0.4,
                       'prob': 0.6
                   }, {
                       'type': 'Contrast',
                       'magnitude': 0.8,
                       'prob': 1.0
                   }],
                   [{
                       'type': 'Equalize',
                       'prob': 0.8
                   }, {
                       'type': 'Equalize',
                       'prob': 0.6
                   }]]
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=384, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(384, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=384),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    train=dict(
        type='ImageNet',
        data_prefix='/home/sjtu/dataset/imagenet/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', size=384, backend='pillow'),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='ImageNet',
        data_prefix='/home/sjtu/dataset/imagenet/val',
        ann_file=None,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(384, -1), backend='pillow'),
            dict(type='CenterCrop', crop_size=384),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        type='ImageNet',
        data_prefix='/home/sjtu/dataset/imagenet/val',
        ann_file=None,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(384, -1), backend='pillow'),
            dict(type='CenterCrop', crop_size=384),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
evaluation = dict(interval=1, metric='accuracy')
paramwise_cfg = dict(
    custom_keys=dict({
        '.cls_token': dict(decay_mult=0.0),
        '.pos_embed': dict(decay_mult=0.0)
    }))
optimizer = dict(
    type='AdamW',
    lr=0.003,
    weight_decay=0.3,
    paramwise_cfg=dict(
        custom_keys=dict({
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0)
        })))
optimizer_config = dict(grad_clip=dict(max_norm=1.0))
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=10000,
    warmup_ratio=0.0001)
runner = dict(type='EpochBasedRunner', max_epochs=300)
checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = 'https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth'
workflow = [('train', 1)]
data_prefix = '/home/sjtu/dataset/imagenet'
work_dir = './work_dirs/vit-384-c'
gpu_ids = range(0, 1)
