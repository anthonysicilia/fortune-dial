#!/bin/sh
for name in 'easy' 'medium' 'hard';
do
    accelerate launch train.py \
        'build:'$name \
        'build:'$name \
        $name'-if' \
        --model 'TinyLlama/TinyLlama-1.1B-Chat-v0.6' \
        --batch_size 4 \
        --generate_during_val 0 \
        --n_epochs 5 \
        --use_lora 1 \
        --gradient_checkpointing 1 \
        --load_in_4bit 1 \
        --prompt_with_averages 1 \
        --lr 1e-4
done
