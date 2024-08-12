#!/bin/sh

# run previously
# python3 -m src.kl 'easy~'$infix'=interp'
# python3 -m src.kl 'medium~'$infix'=interp'
# python3 -m src.kl 'hard~'$infix'=interp'
# > easy~model=meta-llama+Llama-2-7b-chat-hf~lora=32~4bit=1~alg=interp KL: 13825.238615274428
# > medium~model=meta-llama+Llama-2-7b-chat-hf~lora=32~4bit=1~alg=interp KL: 87118.68112087248
# > hard~model=meta-llama+Llama-2-7b-chat-hf~lora=32~4bit=1~alg=interp KL: 13584.374582767485

infix='model=meta-llama+Llama-2-7b-chat-hf~lora=32~4bit=1~alg'
python3 -m src.kl 'easy~'$infix'=explore'
python3 -m src.kl 'medium~'$infix'=explore'
python3 -m src.kl 'hard~'$infix'=explore'
python3 -m src.kl 'easy~'$infix'=exploit'
python3 -m src.kl 'medium~'$infix'=exploit'
python3 -m src.kl 'hard~'$infix'=exploit'
python3 -m src.kl 'easy~'$infix'=if'
python3 -m src.kl 'medium~'$infix'=if'
python3 -m src.kl 'hard~'$infix'=if'

# easy~model=meta-llama+Llama-2-7b-chat-hf~lora=32~4bit=1~alg=explore KL: 695839.7767066954
# easy~model=meta-llama+Llama-2-7b-chat-hf~lora=32~4bit=1~alg=exploit KL: 645551.5069305896
# easy~model=meta-llama+Llama-2-7b-chat-hf~lora=32~4bit=1~alg=if KL: 32511.23188324272

# medium~model=meta-llama+Llama-2-7b-chat-hf~lora=32~4bit=1~alg=explore KL: 41751.08889341354
# medium~model=meta-llama+Llama-2-7b-chat-hf~lora=32~4bit=1~alg=exploit KL: 57484073.30513
# medium~model=meta-llama+Llama-2-7b-chat-hf~lora=32~4bit=1~alg=if KL: 247102.15025469655

# hard~model=meta-llama+Llama-2-7b-chat-hf~lora=32~4bit=1~alg=explore KL: 363244.1677808761
# hard~model=meta-llama+Llama-2-7b-chat-hf~lora=32~4bit=1~alg=exploit KL: 387388.9492154121
# hard~model=meta-llama+Llama-2-7b-chat-hf~lora=32~4bit=1~alg=if KL: 254812.67514452332

# medium~model=meta-llama+Llama-2-7b-chat-hf~lora=32~4bit=1~alg=exploit2 1856.8472802639003
