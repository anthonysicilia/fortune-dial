#!/bin/sh
method='if'
postfix='model=meta-llama+Llama-2-7b-chat-hf~lora=32~4bit=1~alg='$method
for model in 'easy~'$postfix 'medium~'$postfix 'hard~'$postfix;
do
    for name in 'awry' 'bargains' 'casino' 'cmv' 'donations' 'deals' 'deleted' 'supreme';
    do
        for avg in 1;
        do
            for temp in 0.1 0.25 0.5 0.75 1 1.25 1.5 1.75 2
            do
                python3 -m src.predict \
                    $model \
                    'build:'$name \
                    --valname $name'-tempval@temp='$temp \
                    --batch_size 4 \
                    --num_workers 4 \
                    --temp $temp \
                    --just_some 250 \
                    --split 'val' \
                    --prompt_with_averages $avg \
                    --logit_proba 1
            done
        done
    done
done
