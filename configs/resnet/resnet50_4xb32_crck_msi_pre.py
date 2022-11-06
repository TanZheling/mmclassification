_base_ = [
    '../_base_/models/resnet50_crck.py', '../_base_/datasets/crck_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth',
            prefix='backbone',
        ))
)
runner = dict(type='EpochBasedRunner', max_epochs=100)
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
checkpoint_config = dict(interval=20, max_keep_ckpts=20)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='med-ai',
                entity='zlt', 
                name='super_mut_resnet50_4xb64_lr1e-4_100e_crck_apc'
            )
        )
    ]
)