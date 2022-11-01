_base_ = [
    '../_base_/models/resnet18_crck.py', '../_base_/datasets/crck_bs64_tp53.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/home/sjtu/scratch/zltan/mmclassification/medical/resnet18_4xb64_supervised_lr1e-4_tp53/epoch_80.pth',
            prefix='backbone',
        ))
)
runner = dict(type='EpochBasedRunner', max_epochs=20)
optimizer = dict(type='SGD', lr=0.00001, momentum=0.9, weight_decay=0.0001)
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
                name='super_mut_resnet18_4xb64_lr1e-4_100e_crck_tp53_resume'
            )
        )
    ]
)