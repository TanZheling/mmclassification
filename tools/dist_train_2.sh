#!/usr/bin/env bash
CONFIG=$1
GPUS=$2
PORT=${PORT:-29507}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} 
    #\
    # --log-wandb \
    # --wandb-entity zlt --wandb-project med-ai \
    # --wandb-name super_deepsmile_reproduce_resnet18_4xb64_lr1e-6_50e_INpre 
