#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the original graph2nn github repo.

# File modifications and additions by Rowan AI Lab, licensed under the Creative Commons Zero v1.0 Universal
# LICENSE file in the root directory of this source tree.

## Explaining the arguments:
## Tasks: mlp_cifar10, cnn_cifar10, cnn_cifar100, resnet18_tinyimagenet, resnet18_imagenet, efficient_imagenet
## Division: all
## GPU: 1, 8

## usage sample: bash corruptions-inference_tinyimagenet.sh resnet18_imagenet all 1


TASK=${1:-mlp_cifar10}
DIVISION=${2:-best}
GPU=${3:-1}
OUT=${4:-file} # to analyze result, you should change from "stdout" to "file"

if [ "$TASK" = "mlp_cifar10" ]
then
    DATASET='cifar10'
    PREFIX='mlp_bs128_1gpu_layer3'
elif [ "$TASK" = "mlp_cifar10_bio" ]
then
    DATASET='cifar10'
    PREFIX='mlp_bs128_1gpu_layer3'
elif [ "$TASK" = "cnn_cifar10" ]
then
    DATASET='cifar10'
    PREFIX='cnn6_bs1024_8gpu_64d'
elif [ "$TASK" = "cnn_cifar100" ]
then
    DATASET='cifar100'
    PREFIX='cnn6_bs640_1gpu_64d'
elif [ "$TASK" = "resnet18_tinyimagenet" ]
then
    DATASET='tinyimagenet200'
    PREFIX='R-18_tiny_bs256_1gpu'
elif [ "$TASK" = "cnn_imagenet" ]
then
    DATASET='imagenet'
    if [ "$GPU" = 1 ]
    then
    PREFIX='cnn6_bs32_1gpu_64d'
    else
    PREFIX='cnn6_bs256_8gpu_64d'
    fi
elif [ "$TASK" = "resnet18_imagenet" ]
then
    DATASET='imagenet'
    if [ "$GPU" = 1 ]
    then
        PREFIX='R-18_bs450_1gpu'
    else
        PREFIX='R-18_bs512_8gpu'
    fi
elif [ "$TASK" = "resnet34_imagenet" ]
then
    DATASET='imagenet'
    if [ "$GPU" = 1 ]
    then
        PREFIX='R-34_bs32_1gpu'
    elif [ "$GPU" = 4 ]
    then
        PREFIX='R-34_bs256_4gpu'
    else
        PREFIX='R-34_bs256_8gpu'
    fi
elif [ "$TASK" = "resnet34sep_imagenet" ]
then
    DATASET='imagenet'
    if [ "$GPU" = 1 ]
    then
    PREFIX='R-34_bs32_1gpu'
    else
    PREFIX='R-34_bs256_8gpu'
    fi
elif [ "$TASK" = "resnet50_imagenet" ]
then
    DATASET='imagenet'
    if [ "$GPU" = 1 ]
    then
    PREFIX='R-50_bs32_1gpu'
    else
    PREFIX='R-50_bs256_8gpu'
    fi
elif [ "$TASK" = "efficient_imagenet" ]
then
    DATASET='imagenet'
    if [ "$GPU" = 1 ]
    then
    PREFIX='EN-B0_bs64_1gpu_nms'
    else
    PREFIX='EN-B0_bs512_8gpu_nms'
    fi
else
   exit 1
fi

DIR=configs/baselines/${DATASET}/${TASK}/${DIVISION}/*
echo "TASK = ${TASK}, DATASET = ${DATASET}, PREFIX = ${PREFIX}, GPUs = ${GPU}"

(trap 'kill 0' SIGINT;
for CONFIG in $DIR
do
    if echo "$CONFIG" | grep -q "$PREFIX"; then
        if [ "${CONFIG##*.}" = "yaml" ]; then
            CONFIG=${CONFIG##*/}
            CONFIG=${CONFIG%.yaml}
            echo ${CONFIG}
            # run one model at a time
            # Note: with slurm scheduler, one can run multiple jobs in parallel
            python tools/corruptions-inference-tinyimagenet.py --cfg configs/baselines/${DATASET}/${TASK}/${DIVISION}/${CONFIG}.yaml IS_INFERENCE True IS_DDP False OUT_DIR checkpoint/${DATASET}/${TASK}/${DIVISION}/${CONFIG}/inference CHECKPT_DIR checkpoint/${DATASET}/${TASK}/${DIVISION}/${CONFIG} BN.USE_PRECISE_STATS True LOG_DEST $OUT
        fi
    fi
done
)
