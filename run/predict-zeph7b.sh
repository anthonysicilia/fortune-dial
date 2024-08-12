#!/bin/sh
for name in 'awry' 'bargains' 'casino' 'cmv' 'donations' 'deals' 'deleted' 'supreme';
do
    for avg in 0 1;
    do
        python3 -m src.predict \
            'model=HuggingFaceH4+zephyr-7b-beta~trained=0~4bit=1' \
            'build:'$name \
            --valname $name'-val@avg='$avg'~alg=none' \
            --batch_size 4 \
            --num_workers 4 \
            --temp 0.7 \
            --just_some 250 \
            --split 'val' \
            --use_system 1 \
            --prompt_with_averages $avg \
            --max_new_tokens 256

        python3 -m src.predict \
            'model=HuggingFaceH4+zephyr-7b-beta~trained=0~4bit=1' \
            'build:'$name \
            --valname $name'-test@avg='$avg'~alg=none' \
            --batch_size 4 \
            --num_workers 4 \
            --temp 0.7 \
            --just_some 550 \
            --split 'test' \
            --use_system 1 \
            --prompt_with_averages $avg \
            --max_new_tokens 256
    done

    for avg in 'reminder' 'high' 'low';
    do
        python3 -m src.predict \
            'model=HuggingFaceH4+zephyr-7b-beta~trained=0~4bit=1' \
            'build:'$name \
            --valname $name'-test@avg='$avg'~alg=none' \
            --batch_size 4 \
            --num_workers 4 \
            --temp 0.7 \
            --just_some 550 \
            --split 'test' \
            --use_system 1 \
            --prompt_with_averages $avg \
            --max_new_tokens 256
    done
done
