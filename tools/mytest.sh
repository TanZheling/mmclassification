#!/usr/bin/env bash

CONFIG=$1

# 'gaussian_noise' 
for c in 'jpeg_compression' 'shot_noise' \
            'zoom_blur'	'gaussian_noise' 'motion_blur' 'snow'; do
s=5;


python /home/sjtu/scratch/zltan/mmclassification/tools/my_tent_test.py \
 /home/sjtu/scratch/zltan/mmclassification/mywork/vit-384.py https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth \
 --corruption $c --severity $s --out . --metrics accuracy --wandb-name vit$c --wandb-project vit384;

done