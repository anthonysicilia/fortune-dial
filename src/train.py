'''
Fine-tunes causal LMs like llama2. Some features supported:

- qlora/lora/full finetune. auto-gptq support is WIP due to numerical instability (see clamp_llamarmsnorm_forward)
- accelerate
- prompt vs. completion loss
- gradient accumulation / gradient checkpointing

Here's an example command to train a joke explanation model:

accelerate launch train.py datasets/train_joke_explain.jsonl datasets/val_joke_explain.jsonl explanation_generation --model meta-llama/Llama-2-$size\-hf --batch_size 4 --lr $lr --generate_during_val 0 --n_epochs 5 --use_lora 1 --load_in_4bit 1 --gradient_checkpointing 1

'''

import argparse
import tqdm
import json
import numpy as np
import os
import transformers
import accelerate
import tempfile
import torch
import random
import subprocess
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

from src.utils import add_shared_args, SequenceDataset, get_prompt, \
    get_data, compute_loss, compute_pseudo_loss, assign_pseudo_labels, \
    MixLayer, policy_grad_loss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('train')
    parser.add_argument('val')
    parser.add_argument('task_name')

    add_shared_args(parser)

    parser.add_argument('--model',
                        default='meta-llama/Llama-2-7b-hf',
                        type=str)

    parser.add_argument('--load_in_4bit',
                        default=0,
                        type=int,
                        help='should we load bitsnbytes 4 bit?')

    parser.add_argument('--prompt_loss_weight',
                        default=0.1,
                        help='tradeoff between prompt modeling/completion modeling',
                        type=float)

    parser.add_argument('--n_epochs',
                        type=int,
                        default=10)

    parser.add_argument('--lr',
                        type=float,
                        default=.00001)

    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help='how many steps for gradient accumulation')

    parser.add_argument('--skip_save',
                        type=int,
                        default=0)

    parser.add_argument('--force_run',
                        type=int,
                        default=0,
                        help='if 1, we will force the run even if the output already exists.')

    parser.add_argument('--use_lora',
                        type=int,
                        default=0,
                        help='if 1, we will use LoRA')

    parser.add_argument('--lora_r',
                        type=int,
                        default=32,
                        help='what rank LoRA should be used?')

    parser.add_argument('--just_val',
                        type=int,
                        default=0,
                        help='if 1 we will just do val. good for debugging.')

    parser.add_argument('--gradient_checkpointing',
                        type=int,
                        default=1,
                        help='if 1 we will use gradient checkpointing which is slower, but better memory')

    parser.add_argument('--generate_during_val',
                        type=int,
                        default=1,
                        help='if 1 we will generate actual samples during validation and print them out.')

    parser.add_argument('--test_memory',
                        type=int,
                        default=0,
                        help='if 1 we will put the longest sequence first, which causes OOM early')

    parser.add_argument('--use_adafactor',
                        type=int,
                        default=0,
                        help='if 1 we will use adafactor instead of adamw')

    parser.add_argument('--train_msize',
                        type=int,
                        default=700,
                        help='amount of instances to use for training from each set for multi data')

    parser.add_argument('--val_msize',
                        type=int,
                        default=150,
                        help='amount of instances to use for val from each set for multi data')

    # calibration specific args
    parser.add_argument('--pseudo_labels',
                        type=int,
                        default=0,
                        help='Whether to learn to verbalize probas in token space')
    
    parser.add_argument('--off_policy_policy_grad',
                        type=int,
                        default=0,
                        help='Whether to learn to verbalize via off policy policy gradients')

    parser.add_argument('--policy_checkpoint',
                        type=str,
                        default=None,
                        help='SFT checkpoint to use for off policy')
    
    parser.add_argument('--binomial_off_policy',
                        type=int,
                        default=0,
                        help='Use a simple binomial policy (instead of model) for off policy policy grad')
                        
    parser.add_argument('--binomial_off_proba',
                        type=float,
                        default=1,
                        help='Probability with which to use binomial. Good for exploration')
    
    parser.add_argument('--off_policy_correction',
                        type=int,
                        default=0,
                        help='Use instance weights to correct for policy drift between eval / optim.')
    
    parser.add_argument('--policy_ratio_clip',
                        type=float,
                        default=-1,
                        help='Clip instance weights for correcting drift')

    parser.add_argument('--policy_grad',
                        type=int,
                        default=0,
                        help='Whether to learn to verbalize via policy gradients (eval and optim simultaneously)')
    
    parser.add_argument('--temp_explr',
                        type=int,
                        default=0,
                        help='Whether to use temperature to encourage exploration in early epochs of policy grad')
    
    parser.add_argument('--init_ptemp',
                        type=int,
                        default=5.,
                        help='Initial temp for policy grad, if exploring')

    parser.add_argument('--agglom',
                        type=int,
                        default=0,
                        help='Whether to learn to verablize from clusters via agglom')

    parser.add_argument('--kmeans',
                        type=int,
                        default=0,
                        help='Whether to learn to verablize from clusters via kmeans')
    
    parser.add_argument('--subset_clusters',
                        type=int,
                        default=1,
                        help='Cluster data subsets seperately for multi-data')

    parser.add_argument('--cluster_sz',
                        type=int,
                        default=10,
                        help='Cluster size (not exact for kmeans)')

    parser.add_argument('--mixup',
                        type=int,
                        default=0,
                        help='Whether to cluster with mixup')
    
    parser.add_argument('--alpha',
                        type=float,
                        default=1.,
                        help='Alpha in mixup')
    
    # for adding appropriate model tags with system prompt
    parser.add_argument('--logit_proba', type=int, default=0)

    args = parser.parse_args()
    args.val_stat = 'loss'
    args.prompt_delimiter_string = args.prompt_delimiter_string.strip()

    if not args.tokenizer:
        args.tokenizer = args.model

    args.output_path = (args.task_name + '~val{}'.format(args.val_stat) +
        '={:.5f}' + '~model=' + '{}'.format(args.model.replace('/', '+')) +
        '~lora={}'.format(args.use_lora if args.use_lora == 0 else args.lora_r) + '~lr={}'.format(args.lr) +
        '~4bit={}'.format(args.load_in_4bit) + '~promptloss={}'.format(args.prompt_loss_weight) +
        ('.pt' if not args.use_lora else '')
    )
    
    if not args.force_run:
        toks = args.output_path.split('/')
        outdir = '/'.join(toks[:-1]) if len(toks) > 1 else '.'
        def fnameparse(x):
            try:
                return (x.split('~val')[0], '~'.join(x.split('~val')[1].split('~')[1:]))
            except IndexError:
                # path doesn't match, basically skip
                return ''
        existance = set([fnameparse(x) for x in os.listdir(outdir) if '~model=' in x and '~prompt=' not in x])
        if fnameparse(args.output_path) in existance:
            print('{} done already, run with --force_run to run.'.format(args.output_path))
            quit()

    return args

# https://www.reddit.com/r/LocalLLaMA/comments/15sgg4m/what_modules_should_i_target_when_training_using/
def find_target_modules(model):
    # Initialize a Set to Store Unique Layers
    unique_layers = set()
    
    # Iterate Over All Named Modules in the Model
    for name, module in model.named_modules():
        # Check if the Module Type Contains 'Linear4bit'
        if "Linear4bit" in str(type(module)):
            # Extract the Type of the Layer
            layer_type = name.split('.')[-1]
            
            # Add the Layer Type to the Set of Unique Layers
            unique_layers.add(layer_type)

    # Return the Set of Unique Layers Converted to a List
    return list(unique_layers)

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    accelerator = accelerate.Accelerator()
    mainproc = accelerator.is_local_main_process

    if not args.skip_save and mainproc:
        print('saving to {}'.format(args.output_path))

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, 
        padding_side='left')
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    train = get_data(args.train, 'train')
    val = get_data(args.val, 'val')
    prompt = get_prompt(args.val,   proba=args.pseudo_labels,  average=args.prompt_with_averages)

    if args.test_memory:
        # trick: put the very longest example first to check for OOM
        # only do this for single datasets
        assert type(train) == list
        assert type(val) == list
        print('tokenizing all instances to put longest first')
        toked_lengths_train = [
            len(x)
            for x in tokenizer([
                    d['input'] + ' {} '.format(args.prompt_delimiter_string) + d['target'] 
                    + ' ' + tokenizer.eos_token for d in train
                ])['input_ids']
        ]
        toked_lengths_val = [
            len(x)
            for x in tokenizer([
                    d['input'] + ' {} '.format(args.prompt_delimiter_string) + d['target'] 
                    + ' ' + tokenizer.eos_token for d in val
                ])['input_ids']
        ]

        train = zip(train, toked_lengths_train)
        val = zip(val, toked_lengths_val)
        train = list(sorted(train, key=lambda x: -x[1]))
        val = list(sorted(val, key=lambda x: -x[1]))
        m_tr_l = sum(toked_lengths_train) / len(toked_lengths_train)
        m_v_l = sum(toked_lengths_val) / len(toked_lengths_val)
        print('Note: reported lengths do not consider any built prompts! But, tests do...')
        print('train/val mean lengths: {}/{}'.format(m_tr_l, m_v_l))
        print('train/val max lengths: {}/{}'.format(train[0][1], val[0][1]))
        train_rest = train[1:50]
        val_rest = val[1:50]
        # put the biggest first all together in one batch
        # np.random.shuffle(train_rest); np.random.shuffle(val_rest)
        train = [t[0] for t in [train[0]] + train_rest]
        val = [t[0] for t in [val[0]] + val_rest]
    
    train_loader = SequenceDataset(train, tokenizer, args,
        with_label=not args.policy_grad,
        sample_window=True,
        prompt=prompt,
        multi_size=args.train_msize, # default 700 is always balanced for our data, not used if train is not multi
        sample_seed=None # resample everything each epoch for diversity
    )
    
    val_loader = SequenceDataset(val, tokenizer, args,
        with_label=not args.pseudo_labels,
        sample_window=True,
        prompt=prompt,
        multi_size=args.val_msize, # default 150 always balanced for our data, not used if val is not multi
        sample_seed=0 # sample everything identically each epoch for test set consistency
    )

    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.unk_token_id
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    data_collator = transformers.DataCollatorWithPadding(
        tokenizer,
        return_tensors='pt'
    )

    train_loader = torch.utils.data.DataLoader(
        train_loader, shuffle=(False if args.test_memory else True), collate_fn=data_collator, 
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    val_loader = torch.utils.data.DataLoader(
        val_loader, shuffle=False, collate_fn=data_collator, batch_size=args.batch_size, 
        num_workers=args.num_workers
    )

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
                bnb_4bit_quant_type='nf4'), trust_remote_code=True,
                **flash_args)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        
    else:
        if args.use_bfloat:
            model = transformers.AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                trust_remote_code=True)
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)

    print('The model\'s dtype is {}'.format(model.dtype))

    if args.use_lora:
        try:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.lora_r, lora_alpha=16, lora_dropout=0.1
            )
            model = get_peft_model(model, peft_config)
            print('Using default peft recs for target modules')
        except ValueError:
            if 'mistral' in args.model or 'zephyr' in args.model:
                target_modules = ["q_proj", "v_proj"]
                print('Using default peft recs for mistral target modules:', target_modules)
            else:
                target_modules = find_target_modules(model)
                print('Infering target modules for peft:', target_modules)
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.lora_r, lora_alpha=16, lora_dropout=0.1,
                target_modules=target_modules)
            model = get_peft_model(model, peft_config)
        if mainproc:
            model.print_trainable_parameters()

    trainable_params = model.parameters()
    if not args.use_adafactor:
        optim = torch.optim.AdamW(trainable_params, lr=args.lr)
    else:
        optim = transformers.Adafactor(trainable_params, lr=args.lr, scale_parameter=False, relative_step=False, 
            warmup_init=False)

    best_val_acc, best_val_loss, not_improved_epoch = 0.0, np.inf, 0

    if mainproc:
        if args.use_lora:
            tmpfile = tempfile.TemporaryDirectory()
        else:
            tmpfile = tempfile.NamedTemporaryFile()
        print('using tempfile {}'.format(tmpfile.name))

    model, optim, train_loader, val_loader = accelerator.prepare(model, optim, train_loader, val_loader)
    streamer = transformers.TextStreamer(tokenizer)

    for epoch in range(args.n_epochs):
        if mainproc:
            print('Epoch {}'.format(epoch))
        for mode in ['train', 'val']:
            # adding to hopefully resolve some timeout bugs
            accelerator.wait_for_everyone()
            if mode == 'train':
                if args.just_val: continue
                print('Sampling multi data...')
                if train_loader.dataset.multi_data:
                    # epoch seeds ensure processes are on same page, regardless of underlying impl.
                    train_loader.dataset.sample_multi_data(seed=epoch)
                if args.pseudo_labels:
                    print('Assigning pseudo labels...')
                    model.eval()
                    assign_pseudo_labels(train_loader, model, seed=epoch, args=args, accelerator=accelerator)
                model.train()
                bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), disable=not mainproc)
            else:
                model.eval()
                bar = tqdm.tqdm(enumerate(val_loader), total=len(val_loader), disable=not mainproc)

            epoch_stats = {
                'n_batch': 0.0, 'n_exs': 0.0, 'running_sum_loss': 0.0,
                'running_sum_prompt_loss': 0.0, 'running_sum_completion_loss': 0.0,
                'running_sum_prompt_acc': 0.0, 'running_sum_completion_acc': 0.0,
                'running_sum_first_tok_completion_acc': 0.0
            }

            for i, batch in bar:
                if args.just_some > 0 and i * args.batch_size > args.just_some:
                    break
                with torch.set_grad_enabled(mode=='train'):
                    if args.pseudo_labels and (mode == 'val'):
                        prompt_loss = torch.tensor(0.)
                        sample = accelerator.unwrap_model(model).generate(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False
                        )
                        sample = [tokenizer.decode(s, skip_special_tokens=True).strip() for s in sample]
                        targets = [
                            val_loader.dataset.get_target(idx) 
                            for idx in batch['instance_idx'].cpu().numpy()
                        ]
                        completion_loss = compute_pseudo_loss(sample, targets)
                        stats = {'acc_prompt' : float('nan'), 'acc_completion' : float('nan'), 
                            'n_inst_with_correct_first_tok_completion' : float('nan')
                        }
                        stats = {k : torch.tensor(v) for k,v in stats.items()}
                        n_exs = len(sample)
                    elif args.pseudo_labels and args.policy_grad and (mode == 'train'):
                        prompt_loss = torch.tensor(0.)
                        n_exs = batch['input_ids'].size(0)
                        stats = {'acc_prompt' : float('nan'), 'acc_completion' : float('nan'), 
                            'n_inst_with_correct_first_tok_completion' : float('nan')
                        }
                        stats = {k : torch.tensor(v) for k,v in stats.items()}
                        instances = batch['instance_idx'].cpu().numpy()
                        targets = [train_loader.dataset.get_target(idx) for idx in instances]
                        policy_temp = 1.
                        if args.temp_explr:
                            # start at init, linear decrease to 1 halfway through training, stay at 1 rest of time
                            decay = 2 * (epoch + i / len(bar)) / args.n_epochs
                            policy_temp = max(args.init_ptemp * (1 - decay), policy_temp)
                        prompt_loss, completion_loss, stats = policy_grad_loss(model, tokenizer, batch, targets, accelerator=accelerator, temp=policy_temp)
                    elif args.pseudo_labels and args.off_policy_policy_grad and (mode == 'train'):
                        output = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                        prompt_loss, completion_loss, stats = compute_loss(batch['input_ids'], batch['attention_mask'], 
                            batch['prompt_ends_idx'], output['logits'], weights=batch['weights'],
                            off_policy_correction=args.off_policy_correction,
                            policy_ratio_clip=args.policy_ratio_clip)
                        n_exs = output['logits'].shape[0]
                    elif args.mixup and (mode == 'train'):
                        # NOTE: mixup is not tested as multi-gpu yet
                        n_exs = batch['input_ids'].size(0)
                        try:
                            mix = {
                                'pairs' : torch.tensor(
                                    [random.choice([j for j in range(n_exs) if j != i]) for i in range(n_exs)], 
                                    device=batch['attention_mask'].device
                                ).long(),
                                'ratios' : torch.tensor(
                                    [np.random.beta(args.alpha, args.alpha) for _ in range(n_exs)], 
                                    device=batch['attention_mask'].device
                                )
                            }
                        except IndexError:
                            # 1 element batch (at end) - mixup here does nothing
                            # for pseudo labels skip batch to avoid 100% probability assignment
                            if args.pseudo_labels:
                                continue
                            # for normal training, treat this like normal MLE
                            mix = {
                                'pairs' : torch.tensor(
                                    [random.choice([j for j in range(n_exs)]) for i in range(n_exs)], 
                                    device=batch['attention_mask'].device
                                ).long(),
                                'ratios' : torch.tensor(
                                    [np.random.beta(args.alpha, args.alpha) for _ in range(n_exs)], 
                                    device=batch['attention_mask'].device
                                )
                            }
                        mix['ratios'] = mix['ratios'].unsqueeze(-1).unsqueeze(-1)
                        layers = accelerator.unwrap_model(model).model.model.layers
                        l = random.choice(range(1, len(layers)))
                        layers[l] = MixLayer(mix, layers[l])

                        # extra work to get labels for proba verbalization, if needed
                        if args.pseudo_labels:
                            cdevice = accelerator.device # batch['attention_mask'].device
                            instances = batch['instance_idx'].cpu().numpy()
                            targets = [train_loader.dataset.get_target(idx) for idx in instances]
                            targets = torch.tensor(list(map(lambda t: 1 if 'A' in t else 0, targets)), 
                                device=cdevice)
                            lam = mix['ratios'].squeeze()
                            targets = 100 * (lam * targets + (1 - lam) * targets[mix['pairs']]).squeeze()
                            for idx, t in zip(instances, targets.cpu().tolist()):
                                train_loader.dataset.update_pseudo_label(idx, f'{round(t)}%')
                            train_loader.dataset.pseudo_labels = True
                            # remake batch after mixup pseudo labeling - this is definitely inefficient...
                            batch = train_loader.collate_fn([train_loader.dataset.__getitem__(idx) for idx in instances])
                            # funny case, since we don't use dataloader as ususal here, accelerate doesn't send to device
                            batch['input_ids'] = batch['input_ids'].to(cdevice)
                            batch['attention_mask'] = batch['attention_mask'].to(cdevice)
                            batch['prompt_ends_idx'] = batch['prompt_ends_idx'].to(cdevice)
                            # is pseudo labels, reset mix so compute_loss predicts the exact tokens
                            mix = None
                        output = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                        # should be able to reset things right away, pytorch will remember the computation graph
                        layers[l] = layers[l].layer
                        prompt_loss, completion_loss, stats = compute_loss(batch['input_ids'], batch['attention_mask'], 
                            batch['prompt_ends_idx'], output['logits'], mix=mix)
                    else:
                        output = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                        prompt_loss, completion_loss, stats = compute_loss(batch['input_ids'], batch['attention_mask'], 
                            batch['prompt_ends_idx'], output['logits'])
                        n_exs = output['logits'].shape[0]

                    loss = args.prompt_loss_weight * prompt_loss.mean() + completion_loss.mean()
                    if mode == 'train':
                        loss_scaled = loss / args.gradient_accumulation_steps
                        accelerator.backward(loss_scaled)
                        if i % args.gradient_accumulation_steps == 0 or i == len(train_loader) - 1:
                            optim.step()
                            optim.zero_grad()
                    if mainproc:
                        epoch_stats['n_batch'] += 1
                        epoch_stats['n_exs'] += n_exs
                        epoch_stats['running_sum_loss'] += loss.cpu().detach().numpy()
                        epoch_stats['running_sum_prompt_loss'] += prompt_loss.cpu().detach().numpy()
                        epoch_stats['running_sum_completion_loss'] += completion_loss.cpu().detach().numpy()
                        epoch_stats['running_sum_prompt_acc'] += stats['acc_prompt'].cpu().numpy()
                        epoch_stats['running_sum_completion_acc'] += stats['acc_completion'].cpu().numpy()
                        epoch_stats['running_sum_first_tok_completion_acc'] += stats['n_inst_with_correct_first_tok_completion'].cpu().numpy()

                        bar.set_description('loss = {:.6f} (loss prompt/completion = {:.3f}/{:.3f}; token acc prompt/completion/clf first tok = {:.2f}%/{:.2f}%/{:.2f}%)'.format(
                            epoch_stats['running_sum_loss'] / epoch_stats['n_batch'],
                            epoch_stats['running_sum_prompt_loss'] / epoch_stats['n_batch'],
                            epoch_stats['running_sum_completion_loss'] / epoch_stats['n_batch'],
                            100*epoch_stats['running_sum_prompt_acc'] / epoch_stats['n_batch'],
                            100*epoch_stats['running_sum_completion_acc'] / epoch_stats['n_batch'],
                            100*epoch_stats['running_sum_first_tok_completion_acc'] / epoch_stats['n_exs']))

                        if mode == 'val' and args.generate_during_val:
                            print(tokenizer.decode(batch['input_ids'][0][:batch['prompt_ends_idx'][0]]))
                            sample = accelerator.unwrap_model(model).generate(
                                input_ids=batch['input_ids'][:1][:, :batch['prompt_ends_idx'][0]],
                                max_new_tokens=128, streamer=streamer
                            )
                            print('prediction: {}'.format(tokenizer.decode(sample[0][batch['prompt_ends_idx'][0]:]).strip()))
                            print('~'*10)

            if mode == 'val' and mainproc:
                val_loss = epoch_stats['running_sum_loss'] / epoch_stats['n_batch']
                print('we computed accuracy/loss over {} validation examples.'.format(epoch_stats['n_exs']))
                best_yet = val_loss < best_val_loss

                if best_yet:
                    print('{} is a better than than {}, saving weights!'.format(
                        val_loss,
                        best_val_loss))
                    best_val_loss = val_loss
                    if not args.skip_save:
                        if not args.use_lora:
                            torch.save(
                                {'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                                 'args': vars(args)},
                                tmpfile.name)
                        else:
                            accelerator.unwrap_model(model).save_pretrained(tmpfile.name)
                        not_improved_epoch = 0
                else:
                    not_improved_epoch += 1

            if args.just_val: break

    accelerator.wait_for_everyone()
    if mainproc and not args.skip_save:
        args.output_path = args.output_path.format(best_val_loss)
        subprocess.call('cp -R {} {}'.format(tmpfile.name, args.output_path), shell=True)


if __name__ == '__main__':
    main()
