#!/bin/sh
lr=1e-4
sz=10
for name in 'easy' 'medium' 'hard';
do
    accelerate launch train.py \
        'build:'$name \
        'build:'$name \
        $name'-l-'$sz \
        --model 'meta-llama/Llama-2-7b-chat-hf' \
        --batch_size 3 \
        --num_workers 3 \
        --generate_during_val 0 \
        --n_epochs 5 \
        --use_lora 1 \
        --gradient_checkpointing 1 \
        --load_in_4bit 1 \
        --prompt_with_averages 1 \
        --pseudo_labels 1 \
        --cluster_sz $sz \
        --kmeans 1 \
        --lr $lr
done
