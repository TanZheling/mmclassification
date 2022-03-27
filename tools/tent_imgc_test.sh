#!/usr/bin/env bash

CONFIG=$1

# 'gaussian_noise' 
for c in 'defocus_blur'	'pixelate' 'glass_blur' \
	        'elastic_transform'	'brightness' 'fog' 'contrast' \
    	    'frost'	'impulse_noise'	'jpeg_compression' 'shot_noise' \
            'zoom_blur'	'gaussian_noise' 'motion_blur' 'snow'; do
s=5;


python /run/determined/workdir/scratch/mmclassification/tools/my_tent_test.py \
 $CONFIG /run/determined/workdir/scratch/bishe/pretrained_model/INTERN_models/vit-b.pth \
 --corruption $c --severity $s --out . --metrics accuracy --wandb-name $c ;

done