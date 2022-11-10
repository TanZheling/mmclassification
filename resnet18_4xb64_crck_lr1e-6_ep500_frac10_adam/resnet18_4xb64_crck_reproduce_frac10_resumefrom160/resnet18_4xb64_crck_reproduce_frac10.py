model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth',
            prefix='backbone')),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 2)))
dataset_type = 'Crck'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    train=dict(
        type='Crck',
        data_prefix='/home/dqwang/colon_train_10',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='Crck',
        data_prefix='/home/dqwang/CRC_DX_test',
        ann_file=None,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        type='Crck',
        data_prefix='/home/dqwang/CRC_DX_test',
        ann_file=None,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
evaluation = dict(interval=1, metric='accuracy')
optimizer = dict(type='Adam', lr=1e-06, betas=(0.9, 0.99), weight_decay=0.0001)
optimizer_config = dict()
lr_config = dict(policy='fixed')
runner = dict(type='EpochBasedRunner', max_epochs=500)
checkpoint_config = dict(interval=10, max_keep_ckpts=20)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='med-ai',
                entity='zlt',
                name=
                'super_deepsmile_reproduce_resnet18_4xb64_lr1e-6_500e_INpre_frac10'
            ))
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = '/home/dqwang/mmclassification/medical/resnet18_4xb64_crck_lr1e-6_ep500_frac10_adam/epoch_160.pth'
workflow = [('train', 1)]
vote = True
work_dir = 'resnet18_4xb64_crck_lr1e-6_ep500_frac10_adam/resnet18_4xb64_crck_reproduce_frac10_resumefrom160'
gpu_ids = range(0, 4)
