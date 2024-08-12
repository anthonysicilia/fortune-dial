#!/bin/sh
for name in 'easy' 'medium' 'hard';
do
    accelerate launch train.py \
        'build:'$name \
        'build:'$name \
        $name'-if-long' \
        --model 'meta-llama/Llama-2-7b-chat-hf' \
        --batch_size 4 \
        --generate_during_val 0 \
        --n_epochs 5 \
        --use_lora 1 \
        --gradient_checkpointing 1 \
        --load_in_4bit 1 \
        --prompt_with_averages 1 \
        --lr 1e-4 \
        --train_msize 5000
done
