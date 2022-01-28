_base_=['../configs/vision_transformer/vit-base-p16_ft-64xb64_in1k-384.py']
dataset_type = 'ImageNet'
data=dict(
    samples_per_gpu=64,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_prefix='data/imagenet/train',
        ),
    val=dict(
        type=dataset_type,
        data_prefix='data/imagenet-c/brightness/1',
        ann_file=None,
        ),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='data/imagenet-c/brightness/1',
        ann_file=None,
        )
)
