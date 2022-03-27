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