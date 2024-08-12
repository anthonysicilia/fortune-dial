#!/bin/sh
eps=0.5
lr=1e-4
for name in 'easy' 'hard';
do
    accelerate launch train.py \
        'build:'$name \
        'build:'$name \
        $name'-s-'$eps \
        --model 'meta-llama/Llama-2-7b-chat-hf' \
        --batch_size 4 \
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
        --policy_checkpoint 'medium-if~valloss=0.54460~model=meta-llama+Llama-2-7b-chat-hf~lora=32~lr=0.0001~4bit=1~promptloss=0.1' \
        --lr $lr
done
