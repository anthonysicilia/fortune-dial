#!/bin/sh
name='final'

accelerate launch train.py \
    'build:'$name \
    'build:'$name \
    $name'-if-long' \
    --model 'meta-llama/Meta-Llama-3.1-8B-Instruct' \
    --batch_size 3 \
    --num_workers 3 \
    --generate_during_val 0 \
    --n_epochs 5 \
    --use_lora 1 \
    --gradient_checkpointing 1 \
    --load_in_4bit 1 \
    --prompt_with_averages 1 \
    --lr 1e-4 \
    --train_msize 5000

name='final'
lr=1e-4
eps=0.5
sz=10

accelerate launch train.py \
    'build:'$name \
    'build:'$name \
    $name'-c-'$eps'-'$sz'-long' \
    --model 'meta-llama/Meta-Llama-3.1-8B-Instruct' \
    --batch_size 3 \
    --num_workers 3 \
    --generate_during_val 0 \
    --n_epochs 5 \
    --use_lora 1 \
    --gradient_checkpointing 1 \
    --load_in_4bit 1 \
    --prompt_with_averages 1 \
    --pseudo_labels 1 \
    --off_policy_policy_grad 1 \
    --off_policy_correction 1 \
    --policy_ratio_clip $eps \
    --cluster_sz $sz \
    --kmeans 1 \
    --lr $lr \
    --train_msize 5000
