#!/bin/sh
method='if'
postfix='model=meta-llama+Llama-2-7b-chat-hf~lora=32~4bit=1~alg='$method
for model in 'easy~'$postfix 'medium~'$postfix 'hard~'$postfix;
do
    for name in 'awry' 'bargains' 'casino' 'cmv' 'donations' 'deals' 'deleted' 'supreme';
    do
        for avg in 0 1 'reminder' 'high' 'low';
        do
            python3 -m src.predict \
                $model \
                'build:'$name \
                --valname $name'-excess@avg='$avg'~alg='$method \
                --batch_size 4 \
                --num_workers 4 \
                --temp 1 \
                --just_some 150 \
                --split 'test' \
                --prompt_with_averages $avg \
                --logit_proba 2
        done
    done
done
