#!/bin/sh
name='medium'

for lr in 1e-5 5e-5 1e-4;
do
   accelerate launch train.py \
       'build:'$name \
       'build:'$name \
       $name'-if' \
       --model 'meta-llama/Llama-2-7b-chat-hf' \
       --batch_size 4 \
       --generate_during_val 0 \
       --n_epochs 5 \
       --use_lora 1 \
       --gradient_checkpointing 1 \
       --load_in_4bit 1 \
       --prompt_with_averages 1 \
       --lr $lr
done