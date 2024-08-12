#!/bin/sh
for name in 'deleted';
do
    for avg in 'high';
    do
        python3 -m src.predict \
            'model=openai+gpt-4~trained=0' \
            'build:'$name \
            --valname $name'-test@avg='$avg'~alg=none' \
            --batch_size 1 \
            --num_workers 0 \
            --temp 1 \
            --just_some 550 \
            --split 'test' \
            --use_system 1 \
            --prompt_with_averages $avg \
            --max_new_tokens 256
    done
done
