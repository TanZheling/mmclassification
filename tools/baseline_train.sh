#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29501}

for c in 'defocus_blur'; do
s=5;
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/baseline_train.py $CONFIG --launcher pytorch ${@:3} --corruption $c --level $s;

done