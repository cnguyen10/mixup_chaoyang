#!/bin/sh
DEVICE_ID=0  # which GPU is going to be used
export CUDA_VISIBLE_DEVICES=${DEVICE_ID}

BASEDIR=$(cd $(dirname $0) && pwd)

python3 -W ignore "$BASEDIR/main.py" \
    --experiment-name "Mixup" \
    --run-description "" \
    --train-file "/sda2/datasets/chaoyang/train_multiple_labels.json" \
    --test-file "/sda2/datasets/chaoyang/test.json" \
    --num-classes "4" \
    --lr "0.01" \
    --batch-size "256" \
    --num-epochs "300" \
    --jax-platform "cuda" \
    --mem-frac "0.9" \
    --no-majority-vote \
    --alpha 0.4 \
    --mixup \
    # --run-id "c0b82cf1a207452db03a6cf66a8ebeaa"
    # --pretrained-params-path "/sda2/pretrained_models/resnet18_weights.h5"