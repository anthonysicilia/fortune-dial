#!/bin/sh
for name in 'deleted';
do
    for avg in 0 1;
    do
        python3 -m src.predict \
            'model=meta-llama+Llama-2-70b-chat-hf~trained=0~4bit=1' \
            'build:'$name \
            --valname $name'-val@avg='$avg'~alg=none' \
            --batch_size 1 \
            --num_workers 0 \
            --temp 0.6 \
            --top_p 0.9 \
            --just_some 250 \
            --split 'val' \
            --use_system 1 \
            --prompt_with_averages $avg \
            --max_new_tokens 256
        
        python3 -m src.predict \
            'model=meta-llama+Llama-2-70b-chat-hf~trained=0~4bit=1' \
            'build:'$name \
            --valname $name'-test@avg='$avg'~alg=none' \
            --batch_size 1 \
            --num_workers 0 \
            --temp 0.6 \
            --top_p 0.9 \
            --just_some 550 \
            --split 'test' \
            --use_system 1 \
            --prompt_with_averages $avg \
            --max_new_tokens 256
    done

    for avg in 'reminder' 'high' 'low';
    do
        python3 -m src.predict \
            'model=meta-llama+Llama-2-70b-chat-hf~trained=0~4bit=1' \
            'build:'$name \
            --valname $name'-test@avg='$avg'~alg=none' \
            --batch_size 1 \
            --num_workers 0 \
            --temp 0.6 \
            --top_p 0.9 \
            --just_some 550 \
            --split 'test' \
            --use_system 1 \
            --prompt_with_averages $avg \
            --max_new_tokens 256
    done
done
