#!/usr/bin/env bash

CONFIG=$1

# 'gaussian_noise' 
for c in 'defocus_blur'	'pixelate' 'glass_blur' \
	        'elastic_transform'	'brightness' 'fog' 'contrast' \
    	    'frost'	'impulse_noise'	'jpeg_compression' 'shot_noise' \
            'zoom_blur'	'gaussian_noise' 'motion_blur' 'snow'; do
s=5;


python /home/sjtu/scratch/zltan/mmclassification/tools/my_tent_test.py \
 /home/sjtu/scratch/zltan/mmclassification/mywork/vit-b.py  /home/sjtu/scratch/zltan/pretrained_models/timm_models/vit-b.pth \
 --corruption $c --severity $s --out . --metrics accuracy --wandb-name vit$c --wandb-project usevitbm;

done