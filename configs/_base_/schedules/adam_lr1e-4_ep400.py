optimizer = dict(type='Adam', lr=1e-4, betas=(0.9, 0.99), weight_decay=0)
optimizer_config = dict()  # grad_clip, coalesce, bucket_size_mb
lr_config = dict(
    policy='step',
    step=[420]
    )
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=400)