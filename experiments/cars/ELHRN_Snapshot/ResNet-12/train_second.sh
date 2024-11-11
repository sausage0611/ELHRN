#!/bin/bash
python train_second.py \
    --opt sgd \
    --lr 1e-2 \
    --gamma 1e-1 \
    --epoch 20 \
    --stage 1 \
    --val_epoch 1 \
    --weight_decay 5e-4 \
    --nesterov \
    --train_way 30 \
    --train_shot 5 \
    --train_transform_type 0 \
    --test_transform_type 0 \
    --test_shot 1 5 \
    --gpu 0
