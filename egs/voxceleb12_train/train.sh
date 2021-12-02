#!/bin/bash

export NUM_NODES=1
export NUM_GPUS_PER_NODE=2
export NODE_RANK=0
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))

mkdir -p model log
cd ../..
. ./env.sh
cd -

dataset_file="cfg/Voxceleb.yaml"
model_file="cfg/model.yaml"
training_file="cfg/training.yaml"

python3 -m torch.distributed.launch \
       --nproc_per_node=$NUM_GPUS_PER_NODE \
       --nnodes=$NUM_NODES \
       --node_rank $NODE_RANK \
       ../../tools/train_xtractor.py --dataset $dataset_file --model $model_file --training $training_file
