model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='b',
        img_size=224,
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
            mode='classy_vision'),
        hidden_dim=3072),
    train_cfg=dict(
        augments=dict(
            type='BatchMixup', alpha=0.2, num_classes=1000, prob=1.0)))
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
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='AutoAugment',
        policies=[[{
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
                  }]]),
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
    dict(type='Resize', size=(256, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=512,
    workers_per_gpu=1,
    train=dict(
        type='ImageNet',
        data_prefix='/datasets/imagenet/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', size=224, backend='pillow'),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='AutoAugment',
                policies=[[{
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
                          }]]),
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
        type='ImageNet',
        data_prefix='/datasets/imagenet/val',
        ann_file='/datasets/imagenet/meta/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(256, -1), backend='pillow'),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        type='ImageNet',
        data_prefix='/datasets/imagenet/val',
        ann_file='/datasets/imagenet/meta/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(256, -1), backend='pillow'),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
evaluation = dict(interval=1, metric='accuracy')
optimizer = dict(type='AdamW', lr=0.003, weight_decay=0.3)
optimizer_config = dict(grad_clip=dict(max_norm=1.0))
paramwise_cfg = dict(
    custom_keys=dict({
        '.backbone.cls_token': dict(decay_mult=0.0),
        '.backbone.pos_embed': dict(decay_mult=0.0)
    }))
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=10000,
    warmup_ratio=0.0001)
runner = dict(type='EpochBasedRunner', max_epochs=300)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='transformer',
                entity='zlt',
                name='mmcls-vit-b-p16bs512x4-imagenet'))
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = './work_dirs/vit-b16-bs512x4'
gpu_ids = range(0, 4)
