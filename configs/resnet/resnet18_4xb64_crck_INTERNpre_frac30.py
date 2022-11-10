_base_ = [
    '../_base_/models/resnet18_crck.py', '../_base_/datasets/crck_bs64_frac30.py',
    '../_base_/schedules/adam_lr1e-6_ep50.py', '../_base_/default_runtime.py'
]

runner = dict(type='EpochBasedRunner', max_epochs=200)
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/home/dqwang/pretrain_model/intern_r18.pth',
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
                name='super_deepsmile_reproduce_resnet18_4xb64_lr1e-6_200e_intern_frac30'
            )
        )
    ]
)