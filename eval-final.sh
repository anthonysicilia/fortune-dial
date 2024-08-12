#!/bin/sh
method='final-if-long'
model='final-if-long~valloss=0.29868~model=meta-llama+Meta-Llama-3.1-8B-Instruct~lora=32~lr=0.0001~4bit=1~promptloss=0.1'
for name in 'cmv' 'deleted' 'supreme';
do
    for avg in 0 1;
    do
        CUDA_VISIBLE_DEVICES=0 python3 -m src.predict \
            $model \
            'build:'$name \
            --valname $name'-val@avg='$avg'~alg='$method \
            --batch_size 1 \
            --num_workers 0 \
            --temp 1 \
            --just_some 250 \
            --split 'val' \
            --prompt_with_averages $avg \
            --logit_proba 1

        CUDA_VISIBLE_DEVICES=0 python3 -m src.predict \
            $model \
            'build:'$name \
            --valname $name'-test@avg='$avg'~alg='$method \
            --batch_size 1 \
            --num_workers 0 \
            --temp 1 \
            --just_some 550 \
            --split 'test' \
            --prompt_with_averages $avg \
            --logit_proba 1
    done
done
