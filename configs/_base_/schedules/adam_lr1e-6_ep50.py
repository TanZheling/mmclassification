optimizer = dict(type='Adam', lr=1e-6, betas=(0.9, 0.99), weight_decay=1e-4)
optimizer_config = dict()  # grad_clip, coalesce, bucket_size_mb
lr_config = dict(
    policy='fixed',
    )
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=50)