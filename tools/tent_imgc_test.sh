#!/usr/bin/env bash

CONFIG=$1

# 'gaussian_noise' 
for c in 'impulse_noise'	'jpeg_compression' 'shot_noise' \
            'zoom_blur'	'gaussian_noise' 'motion_blur' 'snow'; do
s=5;


python /home/sjtu/scratch/zltan/mmclassification/tools/my_tent_test.py \
 /home/sjtu/scratch/zltan/mmclassification/mywork/swin-trans.py https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window12_384_22kto1k-d59b0d1d.pth \
 --corruption $c --severity $s --out . --metrics accuracy --wandb-name swin$c --wandb-project vit384;

done