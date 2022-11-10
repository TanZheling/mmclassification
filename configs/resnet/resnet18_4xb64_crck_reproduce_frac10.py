_base_ = [
    '../_base_/models/resnet18_crck.py', '../_base_/datasets/crck_bs64_frac10.py',
    '../_base_/schedules/adam_lr1e-6_ep50.py', '../_base_/default_runtime.py'
]
vote = True
runner = dict(type='EpochBasedRunner', max_epochs=500)
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth',
            prefix='backbone',
        ))
)
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
                name='super_deepsmile_reproduce_resnet18_4xb64_lr1e-6_500e_INpre_frac10'
            )
        )
    ]
)