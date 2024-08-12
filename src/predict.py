'''
Inference, only single GPU supported for now.
'''
import warnings
# living dangerously, comment this out to see the warnings
warnings.filterwarnings("ignore")

import argparse
import tqdm
import json
import numpy as np
import transformers
import torch
import torch.nn.functional as F
import random
from peft import PeftModel

from src.utils import add_shared_args, SequenceDataset, \
    get_data, get_prompt, get_tokenizer, compute_loss, \
    OpenAIModelWrapper, DemoModelWrapper

def load_args_from_fname(args):
    if args.checkpoint.endswith('/'):
        fname_to_parse = args.checkpoint[:-1].split('/')[-1].split('.pt')[0]
    else:
        fname_to_parse = args.checkpoint.split('/')[-1].split('.pt')[0]
    toks = fname_to_parse.split('~')
    args.trained = 1 # assume trained by default
    args.prompt_loss_weight = float('nan')
    for t in toks:
        if '=' not in t:
            args.task_name = t
        else:
            k, v = t.split('=')
            if k == 'model':
                args.model = v.replace('+', '/')
            elif k == 'lr':
                args.lr = float(v)
            elif k == '4bit':
                args.load_in_4bit = int(v)
            elif k == 'promptloss':
                args.prompt_loss_weight = float(v)
            elif k == 'lora':
                args.use_lora = int(v)
            elif k == 'trained':
                args.trained = int(v)

def batch_to_device(batch, args):
    res = {}
    for k, v in batch.items():
        res[k] = v.to(args.device)
    return res

def make_output_fname(args):
    output_f = str(args.checkpoint)
    if output_f.endswith('.pt'): output_f = output_f[:-3]
    if output_f.endswith('/'): output_f = output_f[:-1]
    output_f = output_f.split('/')[-1]
    output_f += '~preds_for_instances={}'.format(args.valname.split('/')[-1].replace('.jsonl', ''))
    output_f += '.json'
    args.output_f = f'outputs/{output_f}'
    print('saving predictions to {}'.format(args.output_f))

def make_probs_output_fname(args):
    output_f = str(args.checkpoint)
    if output_f.endswith('.pt'): output_f = output_f[:-3]
    if output_f.endswith('/'): output_f = output_f[:-1]
    output_f = output_f.split('/')[-1]
    output_f += '~probs_for_instances={}'.format(args.valname.split('/')[-1].replace('.jsonl', ''))
    output_f += '.json'
    args.output_f = f'outputs/{output_f}'
    print('saving predictions to {}'.format(args.output_f))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('val')

    add_shared_args(parser)
    parser.add_argument('--valname', type=str, default=None,
        help='Name to use (for instances) when saving - to disambiguate runs w/ different args.')
    parser.add_argument('--temp', default=1, type=float)
    parser.add_argument('--top_p', default=1, type=float)
    parser.add_argument('--logit_proba', default=0, type=int)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--use_cache', default=1, type=int)
    parser.add_argument('--run_with_label', default=0, type=int)
    # only needed if trying to do inference with an openai model
    parser.add_argument('--api_key', type=str, default=None,
        help='OpenAI API key.')

    args = parser.parse_args()

    if args.valname is None:
        args.valname = args.val.split('build:')[-1]

    if args.api_key is None:
        try:
            with open('keychain.json', 'r') as inpt:
                keychain = json.load(inpt)
                args.api_key = keychain['openai']['ai2']
        except FileNotFoundError:
            print('OpenAI API key not provided and no keychain found.')

    load_args_from_fname(args)
    if args.logit_proba:
        make_probs_output_fname(args)
    else:
        make_output_fname(args)

    if not args.tokenizer:
        args.tokenizer = args.model

    return args

def main():

    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = get_tokenizer(args.tokenizer)
    # tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
            
    val = get_data(args.val, args.split)

    if args.prompt_with_averages == 'reminder':
        args.prompt_with_averages = 0
        args.prompt_with_reminder = True
        
    prompt = get_prompt(args.val, proba=not args.logit_proba, average=args.prompt_with_averages,
        reminder=args.prompt_with_reminder)

    partial_args = {
        'sample_window' : True,
        'sample_seed' : 0,
        'prompt' : prompt
    }

    val_loader_with_label = SequenceDataset(val, tokenizer, args, 
        **partial_args)
    val_loader_without_label = SequenceDataset(val, tokenizer, args, 
        with_label=False, **partial_args)
    
    # for decoding later
    delimiters = set(val_loader_without_label.prompt_delimiter_string)
    assert len(delimiters) == 1 # prediction only supported for single sets
    prompt_delimiter_string = list(delimiters)[0]

    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.unk_token_id
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    data_collator = transformers.DataCollatorWithPadding(
        tokenizer,
        return_tensors='pt'
    )

    val_loader_with_label = torch.utils.data.DataLoader(
        val_loader_with_label, shuffle=False, collate_fn=data_collator, 
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    val_loader_without_label = torch.utils.data.DataLoader(
        val_loader_without_label, shuffle=False, collate_fn=data_collator, 
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    if 'Meta-Llama-3.1' in args.model:
        llama_3p1_tkns = True
    else:
        llama_3p1_tkns = False

    if 'openai' in args.model:
        model = OpenAIModelWrapper(tokenizer, args)
    elif 'demo_model' in args.model:
        # just print out all the examples
        model = DemoModelWrapper(tokenizer, args)
    elif not args.trained:
        if args.load_in_4bit:

            if args.flash_attn:
                flash_args = {'use_flash_attention_2' : True}
            else:
                flash_args = {}

            model = transformers.AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16 if args.use_bfloat else torch.float16,
                quantization_config=transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=torch.bfloat16 if args.use_bfloat else torch.float16,
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_quant_type='nf4'),
                    **flash_args, trust_remote_code=True)
            print('loaded quantized model')
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    else:
        if args.load_in_4bit:

            if args.flash_attn:
                flash_args = {'use_flash_attention_2' : True}
            else:
                flash_args = {}

            model = transformers.AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16 if args.use_bfloat else torch.float16,
                quantization_config=transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=torch.bfloat16 if args.use_bfloat else torch.float16,
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_quant_type='nf4'),
                    **flash_args, trust_remote_code=True)
            print('loaded quantized model')
        else:
            if args.use_bfloat:
                model = transformers.AutoModelForCausalLM.from_pretrained(args.model, 
                    torch_dtype=torch.bfloat16)
            else:
                model = transformers.AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)

        if args.trained:
            # need to unpack the trained weights
            if not args.use_lora:
                state = torch.load(args.checkpoint)
                state['model_state_dict'] = {
                    k.replace('module.', '') : v 
                    for k, v in state['model_state_dict'].items()
                }
                model.load_state_dict(state['model_state_dict'])
            else:
                model = PeftModel.from_pretrained(model, args.checkpoint)

    print('The model\'s dtype is {}'.format(model.dtype))

    try:
        model.to(args.device)
    except ValueError:
        print('Tried to call .to on 4-bit model')
    model.eval()

    bar = tqdm.tqdm(
        enumerate(zip(val_loader_with_label, val_loader_without_label)), 
        total=len(val_loader_with_label)
    )

    epoch_stats = {
        'n_batch': 0.0, 'n_exs': 0.0, 'running_sum_loss': 0.0,
        'running_sum_prompt_loss': 0.0, 'running_sum_completion_loss': 0.0,
        'running_sum_prompt_acc': 0.0, 'running_sum_completion_acc': 0.0,
        'running_sum_first_tok_completion_acc': 0.0
    }

    instanceid2pred = {}
    for i, (with_label, without_label) in bar:

        if args.just_some > 0 and i * args.batch_size > args.just_some:
            break
            
        with torch.no_grad():

            with_label = batch_to_device(with_label, args)
            without_label = batch_to_device(without_label, args)
            ids = [val_loader_with_label.dataset.get_instance_id(idx) 
                for idx in  with_label['instance_idx'].cpu().numpy()]
            
            if args.run_with_label:
                output = model(input_ids=with_label['input_ids'], 
                    attention_mask=with_label['attention_mask'])

                prompt_loss, completion_loss, stats = compute_loss(with_label['input_ids'], 
                    with_label['attention_mask'], with_label['prompt_ends_idx'], output['logits'])
                loss = args.prompt_loss_weight * prompt_loss.mean() + completion_loss.mean()

                epoch_stats['n_batch'] += 1
                epoch_stats['n_exs'] += output['logits'].shape[0]
                epoch_stats['running_sum_loss'] += loss.cpu().detach().numpy()
                epoch_stats['running_sum_prompt_loss'] += prompt_loss.cpu().detach().numpy()
                epoch_stats['running_sum_completion_loss'] += completion_loss.cpu().detach().numpy()
                epoch_stats['running_sum_prompt_acc'] += stats['acc_prompt'].cpu().numpy()
                epoch_stats['running_sum_completion_acc'] += stats['acc_completion'].cpu().numpy()
                epoch_stats['running_sum_first_tok_completion_acc'] += \
                    stats['n_inst_with_correct_first_tok_completion'].cpu().numpy()

                bar.set_description('loss = {:.6f} (loss prompt/completion = {:.3f}/{:.3f}; token acc prompt/completion/clf first tok = {:.2f}%/{:.2f}%/{:.2f}%)'.format(
                    epoch_stats['running_sum_loss'] / epoch_stats['n_batch'],
                    epoch_stats['running_sum_prompt_loss'] / epoch_stats['n_batch'],
                    epoch_stats['running_sum_completion_loss'] / epoch_stats['n_batch'],
                    100*epoch_stats['running_sum_prompt_acc'] / epoch_stats['n_batch'],
                    100*epoch_stats['running_sum_completion_acc'] / epoch_stats['n_batch'],
                    100*epoch_stats['running_sum_first_tok_completion_acc'] / epoch_stats['n_exs']))

            if args.logit_proba:
                output = model(input_ids=without_label['input_ids'], 
                    attention_mask=without_label['attention_mask'])
                # pos_token_idx = tokenizer(' No')['input_ids'][-1] # good sanitycheck
                if llama_3p1_tkns:
                    pos_token_idx = tokenizer(' Yes')['input_ids'][-1] # Llama 3.1
                else:
                    pos_token_idx = tokenizer('Yes')['input_ids'][-1] # Llama 2
                tempered_logits = output['logits'] / args.temp
                # print('|'+tokenizer.decode(torch.argmax(tempered_logits, dim=-1)[:, -1])+'|'); exit()
                probas = F.softmax(tempered_logits, dim=-1)[:, -1][:, pos_token_idx]
                if args.logit_proba == 2:
                    # compute excess instead of proba
                    pos_token_idx = tokenizer('No')['input_ids'][-1]
                    tempered_logits = output['logits'] / args.temp
                    neg_probas = F.softmax(tempered_logits, dim=-1)[:, -1][:, pos_token_idx]
                    other_probas = 1 - (probas + neg_probas)
                    # get excess proba assigned to other tokens and return this instead
                    probas = other_probas / neg_probas
                # mis-nomer since post processing is the same
                sample = probas.cpu().tolist()
                del tempered_logits; del output; del probas
            elif args.temp == 0:
                sample = model.generate(
                    input_ids=without_label['input_ids'],
                    attention_mask=without_label['attention_mask'],
                    max_new_tokens=args.max_new_tokens,
                    use_cache=bool(args.use_cache),
                    do_sample=False
                )
                # sample = [tokenizer.decode(s, skip_special_tokens=True)\
                #     .split(prompt_delimiter_string)[-1].strip() for s in sample]
                if llama_3p1_tkns:
                    sample = [tokenizer.decode(s, skip_special_tokens=False)\
                        .split(prompt_delimiter_string)[-1].strip() for s in sample]
                else:
                    sample = [tokenizer.decode(s, skip_special_tokens=True)\
                        .split(prompt_delimiter_string)[-1].strip() for s in sample]
            else:
                sample = model.generate(
                    input_ids=without_label['input_ids'],
                    attention_mask=without_label['attention_mask'],
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    use_cache=bool(args.use_cache),
                    temperature=args.temp,
                    top_p=args.top_p
                )
                # Llama 2
                # sample = [tokenizer.decode(s, skip_special_tokens=True)\
                #    .split(prompt_delimiter_string)[-1].strip() for s in sample]
                # Llama 3.1
                if llama_3p1_tkns:
                    sample = [tokenizer.decode(s, skip_special_tokens=False)\
                        .split(prompt_delimiter_string)[-1].strip() for s in sample]
                else:
                    sample = [tokenizer.decode(s, skip_special_tokens=True)\
                        .split(prompt_delimiter_string)[-1].strip() for s in sample]
                # print(sample); exit()
            if 'demo_model' in args.model:
                print('ids:', ids)
                print('targets', [val_loader_without_label.dataset.get_target(idx) for idx in without_label['instance_idx']])

            for cid, s in zip(ids, sample):
                instanceid2pred[cid] = s

    print('saving {} predictions to {}'.format(len(instanceid2pred), args.output_f))
    with open(args.output_f, 'w') as f:
        f.write(json.dumps(instanceid2pred))

if __name__ == '__main__':
    main()
