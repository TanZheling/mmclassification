_base_ = [
    '../configs/_base_/models/convnext/convnext-base.py',
    '../configs/_base_/datasets/imagenetc.py',
    '../configs/_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../configs/_base_/default_runtime.py',
]

# Model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ConvNeXt',
        arch='base',
        out_indices=(3, ),
        drop_path_rate=0.4,
        gap_before_final_norm=True,
        init_cfg=[
            dict(
                type='TruncNormal',
                layer=['Conv2d', 'Linear'],
                std=.02,
                bias=0.),
            dict(type='Constant', layer=['LayerNorm'], val=1., bias=0.),
        ]),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='convnext-b-ent-test',
                entity='zlt', 
                name='convnext-b-fog'
            )
        )
    ]
)


optimizer = dict(lr=4e-3)

custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]