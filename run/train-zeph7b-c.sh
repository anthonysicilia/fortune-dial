#!/bin/sh
lr=1e-4
eps=0.5
sz=10
for name in 'medium' 'hard';
do
    accelerate launch train.py \
        'build:'$name \
        'build:'$name \
        $name'-c-'$eps'-'$sz \
        --model 'HuggingFaceH4/zephyr-7b-beta' \
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
        --lr $lr
done
