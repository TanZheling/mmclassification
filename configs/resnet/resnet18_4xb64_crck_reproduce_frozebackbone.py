_base_ = [
    '../_base_/datasets/crck_bs64.py',
    '../_base_/schedules/adam_lr1e-6_ep50.py', '../_base_/default_runtime.py'
]

runner = dict(type='EpochBasedRunner', max_epochs=50)

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth',
            prefix='backbone',
        ),
        frozen_stages=3,
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 2),
    ))

#optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
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
                name='super_deepsmile_reproduce_resnet18_4xb64_lr1e-6_50e_freezebackbone'
            )
        )
    ]
)