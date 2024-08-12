#!/bin/sh
method='if'
postfix='model=meta-llama+Llama-2-7b-chat-hf~lora=32~4bit=1~alg='$method
for model in 'easy~'$postfix 'medium~'$postfix 'hard~'$postfix;
do
    for name in 'awry' 'bargains' 'casino' 'cmv' 'donations' 'deals' 'deleted' 'supreme';
    do
        for avg in 0 1;
        do
            python3 -m src.predict \
                $model \
                'build:'$name \
                --valname $name'-val@avg='$avg'~alg='$method \
                --batch_size 4 \
                --num_workers 4 \
                --temp 1 \
                --just_some 250 \
                --split 'val' \
                --prompt_with_averages $avg \
                --logit_proba 1

            python3 -m src.predict \
                $model \
                'build:'$name \
                --valname $name'-test@avg='$avg'~alg='$method \
                --batch_size 4 \
                --num_workers 4 \
                --temp 1 \
                --just_some 550 \
                --split 'test' \
                --prompt_with_averages $avg \
                --logit_proba 1
        done

        for avg in 'reminder' 'high' 'low';
        do
            python3 -m src.predict \
                $model \
                'build:'$name \
                --valname $name'-test@avg='$avg'~alg='$method \
                --batch_size 4 \
                --num_workers 4 \
                --temp 1 \
                --just_some 550 \
                --split 'test' \
                --prompt_with_averages $avg \
                --logit_proba 1
        done
    done
done
