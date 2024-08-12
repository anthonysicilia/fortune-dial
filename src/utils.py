import backoff
import json
import torch
import torch.nn.functional as F
import random
import re
import numpy as np
import openai
import transformers

from collections import defaultdict
from math import log
from peft import PeftModel
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.stats import describe
from scipy.stats import binom
from torch.distributions.categorical import Categorical
from tqdm import tqdm

from src.corpora import build, prompt
from src.corpora.utils import PROBA_Q

GPT4_SYSTEM = """You are NegotiationPredictionGPT, an expert language model at predicting the likelihood of outcomes in human language negotiations. You will be given the first part of a conversation between several different speakers with potentially different goals. Use the first part of the conversation to put yourself in the mindset of the speakers and estimate the likelihood of the requested conversation outcome for these particular speakers. Use the keyword "OUTCOME" to report your predicted probability for the outcome of interest, requested by the user. Report the probability as a percentage using the format "OUTCOME = percent". For example, "OUTCOME = 70%\""""

GPT4_SYSTEM_LOGIT_PROBA = """You are NegotiationPredictionGPT, an expert language model at predicting outcomes in human language negotiations. You will be given the first part of a conversation between several different speakers with potentially different goals. Use the first part of the conversation to put yourself in the mindset of the speakers and decide whether the requested conversation outcome will occur for these particular speakers. Answer with "Yes" or "No" first, before offering any explanation. For example, an acceptable response is "Yes, because...\""""

def maybe_str_or_int(arg):
    try:
        return int(arg)  # try convert to int
    except ValueError:
        pass
    if arg in {"high", "low", "reminder"}:
        return arg
    raise argparse.ArgumentTypeError("x must be an int or special string")

def add_shared_args(parser):

    # seed for script / model init
    parser.add_argument('--seed', type=int, default=1,
        help='Seed for script / model init.')

    parser.add_argument('--min_window', type=int, default=2,
        help='Min # of turns in partial dialogue segment')
    parser.add_argument('--max_tokens', type=int, default=2000,
        help='Max # of tokens in full dialogues')
    # default to a small number, trained models shouldn't output much
    parser.add_argument('--max_new_tokens', default=16, type=int,
        help='Max # of new tokens in any generation')

    # basic prompt args
    parser.add_argument('--prompt_delimiter_string', type=str, default=' <SEP>',
        help='what string should delimit the prompt/completion? Overwritten if building data/prompt')
    parser.add_argument('--prompt_start_string', type=str, default='',
        help='what string should start the prompt')
    parser.add_argument('--use_system', default=0, type=int,
        help='if positive and building data/prompt and model supports, add a system prompt')
    parser.add_argument('--prompt_with_averages', default=0, type=maybe_str_or_int,
        help='if climatology model should be provided in the prompt')
    parser.add_argument('--prompt_with_reminder', default=0, type=maybe_str_or_int,
        help='if verbal reminder likilihood should be provided in the prompt')
    
    # basic inference args
    parser.add_argument('--tokenizer',
        default=None,
        help='if the tokenizer and the model are saved separately, then use this to specify just the tokenizer',
        type=str)
    parser.add_argument('--batch_size',
        default=32,
        type=int)
    parser.add_argument('--num_workers',
        default=4,
        type=int)
    
    # other misc args
    parser.add_argument('--use_bfloat', type=int, default=0,
        help='if 1 we will use bfloat16 instead of float16 for quantized models')
    parser.add_argument('--flash_attn', type=int, default=0,
        help='if 1 we will use flash attention')
    parser.add_argument('--just_some', type=int, default=-1,
        help='if positive we will stop iterations after this many instances.')

# @backoff.on_exception(backoff.fibo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def call_chat_gpt(prompt, args):

    openai.api_key = args.api_key

    if args.use_system:
        messages = [
            {"role": "system", "content" : GPT4_SYSTEM},
            {"role": "user", "content": prompt}
        ]
    else:
        messages = [{"role": "user", "content": prompt}]

    completion = completions_with_backoff(
        model=args.model.split('/')[-1],
        temperature=args.temp,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        messages=messages
    )

    return [choice.message['content']
        for choice in completion.choices][0]

def safe_call_chat_gpt(prompt, args):
    try:
        return call_chat_gpt(prompt, args)
    except openai.error.Timeout as e:
        #Handle timeout error, e.g. retry or log
        print(f"OpenAI API request timed out: {e}")
        pass
    except openai.error.APIError as e:
        #Handle API error, e.g. retry or log
        print(f"OpenAI API returned an API Error: {e}")
        pass
    except openai.error.APIConnectionError as e:
        #Handle connection error, e.g. check network or log
        print(f"OpenAI API request failed to connect: {e}")
        pass
    except openai.error.InvalidRequestError as e:
        #Handle invalid request error, e.g. validate parameters or log
        print(f"OpenAI API request was invalid: {e}")
        pass
    except openai.error.AuthenticationError as e:
        #Handle authentication error, e.g. check credentials or log
        print(f"OpenAI API request was not authorized: {e}")
        pass
    except openai.error.PermissionError as e:
        #Handle permission error, e.g. check scope or log
        print(f"OpenAI API request was not permitted: {e}")
        pass
    except openai.error.RateLimitError as e:
        #Handle rate limit error, e.g. wait or log
        print(f"OpenAI API request exceeded rate limit: {e}")
        pass
    return '**API_Error_encountered**'

class DemoModelWrapper:

    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args
        self.dtype = None
    
    def __call__(self, *args, **kwargs):
        raise AttributeError('OpenAI Wrapper can only be used for generation.')
    
    def generate(self, input_ids, **kwargs):
        x = [self.tokenizer.decode(i, skip_special_tokens=True) for i in input_ids]
        sample = []
        for p in x:
            print('+++++++++++++++')
            print('prompt:', p)
        return ['demo' for s in sample]
    
    def to(self, *args, **kwargs):
        return self
    
    def eval(self):
        return self

class OpenAIModelWrapper:

    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args
        self.dtype = None
    
    def __call__(self, *args, **kwargs):
        raise AttributeError('OpenAI Wrapper can only be used for generation.')
    
    def generate(self, input_ids, **kwargs):
        x = [self.tokenizer.decode(i, skip_special_tokens=True) for i in input_ids]
        sample = []
        for p in x:
            s = safe_call_chat_gpt(p, self.args)
            # # api is slow enough to print and watch if you want
            # print(p)
            # print('--------')
            # print(s)
            # print('+++++++++++++++')
            # # end print statements
            sample.append(s)
        return [self.tokenizer(s)['input_ids'] for s in sample]
    
    def to(self, *args, **kwargs):
        return self
    
    def eval(self):
        return self

def get_tokenizer(tokenizer_arg):
    if 'openai' in tokenizer_arg or 'demo_model' in tokenizer_arg:
        # use gpt2 tokenizer as a placeholder for batching ops (tokenization is inverted before sending to the api)
        return transformers.AutoTokenizer.from_pretrained('gpt2', padding_side='left')
    else:
        return transformers.AutoTokenizer.from_pretrained(tokenizer_arg, padding_side='left')

def get_data(loc, split):
    if loc.startswith('build:'):
        name = loc.split(':')[-1]
        # seed and max_len are now enforced in corpora module
        return build(name, split) #, seed=1, max_len=None)
    else:
        with open(loc, 'r') as f:
            return [json.loads(line) for line in f.readlines()]

def get_prompt(loc, proba=False, average=False, reminder=False):
    if loc.startswith('build:'):
        name = loc.split(':')[-1]
        return prompt(name, proba=proba, average=average, reminder=reminder)
    else:
        print('Not building prompt since not building dataset')
        return None

def get_proba_from_string(s, sentinel='', delimiter='OUTCOME'):
    seps = '\.|/| / |' # period, ratio, ratio with space, nothing
    tails = '%| %|' # percent, percent with space, nothing
    s = s.split(delimiter)[-1]
    try:
        # find all occurences after delimiter
        nums = re.findall(f"\d*(?:{seps})\d+(?:{tails})", s)
        # preferences (% then ratio then nothing - always first after delimiter)
        perc_nums = [num for num in nums if '%' in num]
        ratio_nums = [num for num in nums if '/' in num]
        if perc_nums:
            nums = perc_nums[0]
        elif ratio_nums:
            nums = ratio_nums[0]
        else:
            nums = nums[0]
        # parse the prefered number extracted
        if '/' in nums:
            nums = nums.split('/')
            proba = float(nums[0]) / float(nums[1])
        elif '%' in nums:
            proba = float(nums.split('%')[0]) / 100
        else:
            proba = float(nums)
            if proba > 1:
                proba = proba / 100
        return proba
    except:
        return sentinel

def add_model_tags(prompt, tokenizer, args):
    # if 'llama' in args.model:
    #     head = ''
    #     if args.use_system:
    #         head = '<<SYS>>\n{}\n<</SYS>>'.format(GPT4_SYSTEM)
    #     return '{}\n[INST]\n{}\n[/INST]'.format(head, prompt)
    # else:
    if 'openai' in args.model:
        print('Ignorning model tags for api based model.')
        return prompt
    
    system_prompt = GPT4_SYSTEM
    if args.logit_proba:
        system_prompt = GPT4_SYSTEM_LOGIT_PROBA

    if args.use_system:
        messages = [
            {"role": "system", "content" : system_prompt},
            {"role": "user", "content": prompt}
        ]
    else:
        messages = [{"role": "user", "content": prompt}]
    
    try:
        if not tokenizer.chat_template:
            head = ''
            if args.use_system:
                head = '### System: {}\n\n'.format(system_prompt)
            return '{}### User: {}\n\n### Assistant:'.format(head, prompt)
        else:
            return tokenizer.apply_chat_template(messages, tokenize=False)
    except AttributeError:
        print('Tokenizer has no chat template, ignoring any system prompts')
        return prompt

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, data, tokenizer, args, with_label=True, sample_window=False, sample_seed=None, prompt=None, multi_size=None):
        
        self.tokenizer = tokenizer
        self.args = args
        self.with_label = with_label
        self.sample_window = sample_window
        self.sample_seed = sample_seed
        self.multi_size = multi_size
        # pseudo_labels and importance weights must be set independently, always start false
        self.pseudo_labels = False
        self.importance_weights = False
        self.change_prompt_style = False # only operational for multi-data

        if type(data) == dict:
            # multi dataset
            self.multi_data = data
            self.sample_multi_data(seed=sample_seed)
            # sample_multi_data implicitly builds:
            # - self.data
            # - self.prompt_start_string
            # - self.prompt_delimiter_string

        else:
            self.multi_data = None
            self.data = data

            if prompt is not None:
                start, delimiter = self.get_prompt_from_string(prompt)
            else:
                start = self.args.prompt_start_string
                delimiter = self.args.prompt_delimiter_string
            
            self.prompt_start_string = [start] * len(self.data)
            self.prompt_delimiter_string = [delimiter]* len(self.data)
            self.mean_probas = [sum([int(elem['target'] == 'Yes') for elem in self.data]) / len(self.data)]
        
        if 'openai' not in args.model:
            print('Example Prompt Start:', self.prompt_start_string[0])
            print('Example Prompt Delimiter:', self.prompt_delimiter_string[0])
    
    def get_prompt_from_string(self, prompt):
        prompt = add_model_tags(prompt, self.tokenizer, self.args)
        start = prompt.split('[arg]')[0]
        delimiter = prompt.split('[arg]')[1]
        return start, delimiter
    
    def get_prompt_from_name(self, name, opposite=False):
        name = 'build:{}'.format(name)
        if opposite:
            proba = not self.args.pseudo_labels
        else:
            proba = self.args.pseudo_labels
        prompt = get_prompt(name, proba=proba, average=self.args.prompt_with_averages)
        return self.get_prompt_from_string(prompt)

    def shuffle_in_unison(self, *args, seed=0):
        assert all(len(a) == len(args[0]) for a in args)
        for a in args:
            random.seed(seed)
            random.shuffle(a)

    def sample_multi_data(self, seed=None):

        if seed is not None:
            random.seed(seed)
        
        if self.multi_size is None:
            lens = [len(v) for v in self.multi_data.values()]
            # maintains balance if multi_size is not explicitly set
            multi_size = min(lens)
        else:
            multi_size = self.multi_size

        self.data = []
        self.prompt_start_string = []
        self.prompt_delimiter_string = []
        self.opposite_delimiter_string = []
        self.data_name = []

        for i, k in enumerate(sorted(self.multi_data.keys())):
            domain = self.multi_data[k]
            random.shuffle(domain)
            mi = len(domain[:multi_size])
            self.data.extend(domain[:multi_size])
            start, delimiter = self.get_prompt_from_name(k)
            _, opposite_delimiter = self.get_prompt_from_name(k, opposite=True)
            self.prompt_start_string.extend([start] * mi)
            self.prompt_delimiter_string.extend([delimiter] * mi)
            self.opposite_delimiter_string.extend([opposite_delimiter] * mi)
            self.data_name.extend([i] * mi)

        self.shuffle_in_unison(
            self.data,
            self.prompt_start_string,
            self.prompt_delimiter_string,
            self.opposite_delimiter_string,
            self.data_name,
            seed=seed or 0
        )

        mean_probas = defaultdict(list)
        for name, elem in zip(self.data_name, self.data):
            mean_probas[name].append(int(elem['target'] == 'Yes'))
        self.mean_probas = {k : sum(v) / len(v) for k,v in mean_probas.items()}

    def __getitem__(self, idx):
        input_text = self.data[idx]['input']
        input_start = self.prompt_start_string[idx]
        input_delimiter = self.prompt_delimiter_string[idx]
        if self.change_prompt_style and hasattr(self, 'opposite_delimiter_string'):
            input_delimiter = self.opposite_delimiter_string[idx]
        # print(input_delimiter)

        if self.sample_window:
            if self.sample_seed is not None:
                uiid = int(self.get_instance_id(idx), 16)
                seed = self.sample_seed + uiid
                random.seed(seed)
            input_text = self._sample_window(input_text)
        
        # important during training for indexing
        wspace = ' '
        if self.pseudo_labels:
            target = self.data[idx]['pseudo_target']
        else:
            target = self.data[idx]['target']

        input_text = self._resize(input_text, target,
            input_start, input_delimiter).replace(' || ', '\n')
        
        if input_start:
            input_text = '{}'.format(input_start) + input_text

        if self.tokenizer is None:
            return input_text + '{}'.format(input_delimiter)

        input_seq = self.tokenizer(input_text + '{}{}'.format(input_delimiter, wspace))
        idx_of_sep = len(input_seq['input_ids'])
        if self.with_label:
            seq = input_text + '{}{}'.format(input_delimiter, wspace) + target + ' ' + self.tokenizer.eos_token
            seq = self.tokenizer(seq)
        else:
            # wspace is important to remove for inference, need to double check tokenization :-)
            input_seq = self.tokenizer(input_text + '{}'.format(input_delimiter))
            seq = input_seq

        seq['prompt_ends_idx'] = idx_of_sep - 1
        seq['instance_idx'] = idx

        if hasattr(self, 'data_name'):
            seq['data_name'] = self.data_name[idx]
        
        if self.importance_weights:
            seq['weights'] = self.data[idx]['weight']

        return seq

    def __len__(self):
        return len(self.data)

    def get_instance_id(self, idx):
        return self.data[idx]['instance_id']
    
    def get_target(self, idx):
        return self.data[idx]['target']
    
    def get_mean_proba(self, idx):
        if hasattr(self, 'data_name'):
            name = self.data_name[idx]
        else:
            name = 0
        return self.mean_probas[name]
    
    def update_pseudo_label(self, idx, txt):
        self.data[idx]['pseudo_target'] = txt
    
    def update_importance_weight(self, idx, w):
        self.data[idx]['weight'] = w
    
    def _sample_window(self, input_text):
        inputs = input_text.split(' || ')
        n = len(inputs)
        start = self.args.min_window
        stop = n - 1
        if stop > self.args.min_window:
            k = random.randint(start, stop)
            return ' || '.join(inputs[:k+1])
        else:
            return ' || '.join(inputs)
    
    def _resize(self, input_text, target, start, delimiter):

        tokenizer = self.tokenizer
        if tokenizer is None:
            # resizing only supported when a tokenizer is available
            return input_text
        
        seq = '{} '.format(start) + input_text + '{} '.format(delimiter) + target + ' ' + tokenizer.eos_token
        while len(tokenizer(seq)['input_ids']) > self.args.max_tokens:
            inputs = input_text.split(' || ')
            # first_utt = min([i for i,u in enumerate(inputs) if 'Situation:' not in u]) # situation deprecated
            if inputs[0] == '...':
                del inputs[0]
            inputs[0] = '...'
            input_text = ' || '.join(inputs)
            seq = '{} '.format(start) + input_text + '{} '.format(delimiter) + target + ' ' + tokenizer.eos_token
        
        return input_text

class MixLayer(torch.nn.Module):

    def __init__(self, mix, layer):
        super().__init__()
        self.mix = mix
        self.layer = layer
    
    def __call__(self, x, *args, **kwargs):
        mix = self.mix
        x = mix['ratios'] * x + (1 - mix['ratios']) * x[mix['pairs']]
        return self.layer(x, *args, **kwargs)

# https://discuss.pytorch.org/t/how-could-i-create-one-hot-tensor-while-ignoring-some-label-index/40987/6
def onehot_with_ignore_label(labels, num_class, ignore_label):
    dummy_label = num_class + 1
    mask = labels == ignore_label
    modified_labels = labels.clone()
    modified_labels[mask] = num_class
    # One-hot encode the modified labels
    one_hot_labels = F.one_hot(modified_labels, num_classes=dummy_label)
    # Remove the last row in the one-hot encoding
    one_hot_labels = one_hot_labels[:, :, :-1]
    return one_hot_labels

def compute_pseudo_loss(sample, targets, reduction='mean', tol=1e-9):
    log_probas = []
    # print('sample:', sample)
    # print('targets:', targets)
    for s in sample:

        # NOTE: can't use this, assimptions are not met
        # proba = get_proba_from_string(s)
        # Instead, just grab the last percent proba - it is fine-tuned so this should work
        tails = '%| %|' # percent, percent with space, nothing
        try:
            perc = re.findall(f"\d+(?:{tails})", s)[-1]
            proba = float(perc.strip('%')) / 100
        except IndexError:
            proba = ''

        if proba == '':
            proba = 0.5
        proba = min(1-tol, max(proba, tol))
        log_probas.append((log(proba), log(1 - proba)))

    y = []
    for t in targets:
        if 'Yes' in t:
            y.append(1)
        else:
            y.append(0)
    log_probas = torch.tensor(log_probas)
    y = torch.tensor(y)
    # print(y.size()); print(y); print(log_probas.size()); print(log_probas)
    return torch.nn.CrossEntropyLoss(reduction=reduction)(log_probas, y)

def _policy_grad_environment(tokenizer):
    # causes a fork - unsure how to fix
    vocabulary = [
        f'{x}%'
        for x in range(0, 100)
    ]

    actions = set()
    horizons = []

    for tokens in tokenizer(vocabulary, add_special_tokens=False)['input_ids']:
        horizons.append(len(tokens))
        actions = actions.union(set(tokens))
    
    # sorted to disambiguate indexing behavior of tesnors
    actions = torch.tensor(list(sorted(actions))).long()
    horizon = max(horizons) + 1
    return actions, horizon

def off_policy_policy_eval(model, tokenizer, inputs, targets, accelerator=None):
    actions, horizon = _policy_grad_environment(tokenizer)

    if accelerator is not None:
        unwrapper = lambda m: accelerator.unwrap_model(m)
    else:
        unwrapper = lambda m: m

    with torch.set_grad_enabled(False):
        outputs = unwrapper(model).generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=horizon,
            use_cache=True,
            do_sample=True,
            prefix_allowed_tokens_fn=lambda *args, **kwargs: actions.cpu().tolist(),
            return_dict_in_generate=True,
            output_scores=True
        )
        transition_scores = unwrapper(model).compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )

    input_length = inputs['input_ids'].size(1)
    generated_tokens = outputs.sequences[:, input_length:]
    samples = [tokenizer.decode(t, skip_special_tokens=False) for t in generated_tokens]
    # pseudo loss is negative log liklihood, reward should be maximized (i.e., liklihood)
    reward = -compute_pseudo_loss(samples, targets, reduction='none')
    # print([tokenizer.decode(t, skip_special_tokens=False) for t in generated_tokens])
    # print(transition_scores)
    probas = torch.exp(transition_scores.sum(-1))
    # print('r', reward)
    return reward, samples, probas

def bin_policy_eval(mean_probas, targets, d=20):
    # d draws
    actions = [np.random.binomial(d, mp) for mp in mean_probas]
    probas = [binom.pmf(a, d, mp) for a, mp in zip(actions, mean_probas)]
    samples = [f'{100 * a / 20:.0f}%' for a in actions]
    # pseudo loss is negative log liklihood, reward should be maximized (i.e., liklihood)
    reward = -compute_pseudo_loss(samples, targets, reduction='none')
    return reward, samples, torch.tensor(probas).to(reward.device)

# NOTE: off_policy is legacy arg for when using the same mechanism
def policy_grad_loss(model, tokenizer, inputs, targets, temp=1., accelerator=None):
    # print('temp', temp)
    actions, horizon = _policy_grad_environment(tokenizer)

    if accelerator is not None:
        unwrapper = lambda m: accelerator.unwrap_model(m)
    else:
        unwrapper = lambda m: m

    with torch.set_grad_enabled(False):
        sample_output = unwrapper(model).generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=horizon,
            use_cache=True,
            do_sample=True,
            temperature=temp,
            prefix_allowed_tokens_fn=lambda *args, **kwargs: actions.cpu().tolist(),
        )
        n_new_tokens = sample_output.size(-1) - inputs['attention_mask'].size(-1)
        new_mask = torch.ones(inputs['attention_mask'].size(0), n_new_tokens).to(inputs['attention_mask'].device)
        attention_mask = torch.cat([inputs['attention_mask'], new_mask], dim=-1).long()

    input_length = inputs['input_ids'].size(1)
    generated_tokens = sample_output[:, input_length:]
    samples = [tokenizer.decode(t, skip_special_tokens=False) for t in generated_tokens]
    output = model(input_ids=sample_output, attention_mask=attention_mask)
    reward = -compute_pseudo_loss(samples, targets, reduction='none')
    # negate again to maximize via minimization
    reward = -reward.unsqueeze(-1).to(output['logits'].device)
    return compute_loss(sample_output, attention_mask, inputs['prompt_ends_idx'], output['logits'], weights=reward)

def compute_loss(input_ids, attention_mask, prompt_end_idx, logits, mix=None, weights=None, off_policy_correction=False, policy_ratio_clip=-1):
    '''
    returns prompt_loss and completion_loss
    '''

    loss_fn = torch.nn.CrossEntropyLoss()

    logits = logits[:, :-1]
    targets = input_ids[:, 1:]
    targets_mask = attention_mask[:, 1:]
    # support for left padding
    idx = torch.arange(attention_mask.size(1), 0, -1).to(attention_mask.device)
    first_one = torch.argmax(attention_mask * idx, 1)
    prompt_end_idx = first_one + prompt_end_idx

    # make prompt/completion mask
    idxs = torch.arange(targets_mask.shape[1], device=attention_mask.device).repeat((targets_mask.shape[0], 1))
    is_prompt = (idxs < (prompt_end_idx[:, None]-1)) * 1
    is_completion = (idxs >= (prompt_end_idx[:, None]-1)) * targets_mask
    is_first_tok_completion = (idxs == (prompt_end_idx[:, None]-1)) * targets_mask
    
    targets_prompt = targets * is_prompt + -100 * (1-is_prompt)
    targets_completion = targets * is_completion + -100 * (1-is_completion)

    if mix is not None:
        targets_completion = onehot_with_ignore_label(targets_completion, logits.size(2), -100)
        targets_completion = mix['ratios'] * targets_completion + (1 - mix['ratios']) * targets_completion[mix['pairs']]
        targets_completion = targets_completion.transpose(2, 1)

    # this could probably be refactored to be more efficient
    loss_prompt = loss_fn(logits.transpose(2, 1), targets_prompt)
    if weights is not None:
        w_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        if off_policy_correction: # defaults to true, only time we use weights are for off policy methods currently
            # NOTE: old approach restrict target policy actions space.... not doing this anymore but keeping it for now
            # with torch.no_grad():
            #     actions, _ = _policy_grad_environment(tokenizer)
            #     actions = actions.to(logits.device)
            #     policy_logits = torch.gather(logits, 2, actions.repeat(logits.size(0), logits.size(1), 1))
            #     policy_probas = torch.softmax(policy_logits, 2)
            #     corrections = []
            #     for i, eidx in enumerate(prompt_end_idx):
            #         # drop eos token that is always at the end during training
            #         traj_probas = policy_probas[i, eidx-1:-1]
            #         full_probas = torch.zeros(traj_probas.size(0), logits.size(2)).to(traj_probas.device)
            #         # drop any white space added by .generate or the dataset class by setting probas to 1
            #         full_probas[:, tokenizer(' ', add_special_tokens=False)['input_ids']] = 1
            #         full_probas[:, actions] = traj_probas
            #         # NOTE: useful for debugging
            #         # x = full_probas.index_select(1, targets[i, eidx-1:-1]).diag()
            #         # y = [tokenizer.decode(t, skip_special_tokens=False) for t in targets[i, eidx-1:-1]]
            #         # for xi, yi, m1, m2 in zip(x,y, full_probas.max(1)[0], full_probas.max(1)[1]):
            #         #     print(xi.item(), f'|{yi}|', m1, f'|{tokenizer.decode(m2)}|')
            #         corrections.append(full_probas.index_select(1, targets[i, eidx-1:-1]).diag().prod())
            #     corrections = torch.tensor(corrections).to(weights.device)
            #     # print('final', corrections)
            with torch.no_grad():
                probas = torch.softmax(logits, 2)
                corrections = []
                for i, eidx in enumerate(prompt_end_idx):
                    corrections.append(probas[i, eidx:].index_select(1, targets[i, eidx:]).diag().prod())
                corrections = torch.tensor(corrections).to(weights.device)
                ratios = corrections / weights[:, 1]
                if policy_ratio_clip > 0:
                    ratios = torch.clip(ratios, 1-policy_ratio_clip, 1+policy_ratio_clip)
                weights[:, 0] = ratios * weights[:, 0]
        # NOTE: removed, this step should be done within RL policy eval code now
        # weights = -weights
        weights = weights[:, 0].unsqueeze(-1).unsqueeze(-1)
        loss_completion = (weights * w_loss_fn(logits.transpose(2, 1), targets_completion)).mean()
    else:
        loss_completion = loss_fn(logits.transpose(2, 1), targets_completion)

    # compute some stats and overwrite loss
    with torch.no_grad():
        preds = logits.argmax(2)
        is_token_acc = (preds==targets)
        stats = {
            'acc_prompt': ((is_token_acc*is_prompt).sum(axis=1)*1.0 / (is_prompt.sum(axis=1))).mean(),
            'acc_completion': ((is_token_acc*is_completion).sum(axis=1)*1.0 / (is_completion.sum(axis=1))).mean(),
            'n_inst_with_correct_first_tok_completion': ((is_token_acc*is_first_tok_completion).sum(axis=1)*1.0).sum(), # good for multichoice
            'n_toks_prompt': (1.0*is_prompt).sum(axis=1).mean(),
            'n_toks_completion': (1.0*is_completion).sum(axis=1).mean()
        }

    return loss_prompt, loss_completion, stats

def assign_pseudo_labels(trloader, model, seed, args, accelerator=None):
    if args.mixup or args.policy_grad:
        # set to false, since we didn't set any yet
        trloader.dataset.pseudo_labels = False
        return # if set, pseudo labels will be set in real time for each batch
    elif args.off_policy_policy_grad:
        if args.policy_checkpoint is not None:
            _assign_pretrained_model_pseudo_labels(trloader, model, seed, args, accelerator=accelerator)
        elif args.kmeans:
            # kmeans handles this implicitly
            algo = lambda n_clusters: KMeans(n_clusters, random_state=seed, n_init="auto")
            _assign_clustered_pseudo_labels(trloader, model, seed, args, algo)
        else:
            _assign_off_policy_pseudo_labels(trloader, model, seed, args, accelerator=accelerator)
        trloader.dataset.importance_weights = True # importance weights have been set
    elif args.kmeans:
        algo = lambda n_clusters: KMeans(n_clusters, random_state=seed, n_init="auto")
        _assign_clustered_pseudo_labels(trloader, model, seed, args, algo)
    elif args.agglom:
        algo = lambda n_clusters: AgglomerativeClustering(n_clusters, metric='cosine', linkage='average')
        _assign_clustered_pseudo_labels(trloader, model, seed, args, algo)
    else:
        _assign_random_pseudo_labels(trloader, model, seed, args)
    if accelerator is not None:
        # use same sampler seed so each process has the same pseudo targets that they created
        trloader.iteration -= 1
    trloader.dataset.pseudo_labels = True # pseudo_labels have been set

def _assign_off_policy_pseudo_labels(trloader, model, seed, args, accelerator=None):
    dataset = trloader.dataset
    dataset.sample_seed = seed # set seed so window samples match clustered
    dataset.pseudo_labels = False # pseudo_labels may not be set yet, prevents error from being thrown
    trloader.dataset.importance_weights = False # importance weights may not have been set
    trloader.dataset.with_label = False # ignore labels for policy evaluation
    tokenizer = trloader.dataset.tokenizer
    random.seed(seed)
    for batch in tqdm(trloader):
        instances = batch['instance_idx'].cpu().numpy()
        targets = [trloader.dataset.get_target(idx) for idx in instances]
        if args.binomial_off_policy and (random.random() <= args.binomial_off_proba):
            mean_probas = [trloader.dataset.get_mean_proba(idx) for idx in instances]
            rewards, samples, probas = bin_policy_eval(mean_probas, targets)
        else:
            rewards, samples, probas = off_policy_policy_eval(model, tokenizer, batch, targets, accelerator=accelerator)
        # negate rewards since we always minimize loss downstream
        rewards = -rewards
        for r, s, p, idx in zip(rewards.cpu(), samples, probas.cpu(), instances):
            # print(r, f'|{s}|', p, trloader.dataset.get_target(idx))
            dataset.update_pseudo_label(idx, s.strip())
            dataset.update_importance_weight(idx, [r.item(), p.item()])
    # reset labels
    trloader.dataset.with_label = True

def _assign_random_pseudo_labels(trloader, _, seed, args):
    dataset = trloader.dataset
    clusters = range(len(dataset) // args.cluster_sz)
    counts = defaultdict(list)
    assignments = defaultdict(list)
    random.seed(seed)
    for idx in range(len(dataset)):
        c = random.choice(clusters)
        label = 1 if 'Yes' in dataset.get_target(idx) else 0
        counts[c].append(label)
        assignments[c].append(idx)
    probas = {c : sum(arr) / len(arr) for c, arr in counts.items()}
    for c, arr in assignments.items():
        percent_proba = 100 * probas[c]
        percent_proba = round(percent_proba)
        for idx in arr:
            dataset.update_pseudo_label(idx, f'{percent_proba}%')
    
def _assign_clustered_pseudo_labels(trloader, model, seed, args, cluster_alg):
    # manipulate underlying object
    dataset = trloader.dataset
    dataset.sample_seed = seed # set seed so window samples match clustered

    # get last hidden state
    X = []
    ids = []
    names = []
    print('Pseudo Labels: Gathering hidden states')
    dataset.pseudo_labels = False # pseudo_labels may not be set yet, prevents error from being thrown
    trloader.dataset.importance_weights = False # importance weights may not have been set
    for batch in tqdm(trloader):
        with torch.no_grad():
            output = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
            last_hidden_state = output['hidden_states'][-1].cpu()
            # pool mean over sequence: https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently
            input_mask_expanded = batch['attention_mask'].cpu().unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            X.append(mean_embeddings)
            ids.append(batch['instance_idx'].cpu())
            if 'data_name' in batch:
                names.append(batch['data_name'].cpu())

    print('Pseudo Labels: conducting clustering')
    X = torch.cat(X, dim=0).cpu().numpy()
    ids = torch.cat(ids, dim=0).cpu().tolist()
    # sklearn version, cluster last hidden state, cluster_sz is a rough estimate
    if names and args.subset_clusters:
        print('Pseudo Labels: Multi-data found, performing clustering within each sub-group')
        names = torch.cat(names, dim=0).cpu().numpy()
        clusters = np.zeros_like(names)
        for n in set(names):
            Xn = X[names==n]
            clusters[names==n] = cluster_alg(n_clusters=Xn.shape[0] // args.cluster_sz).fit_predict(Xn)
    else:
        clusters = cluster_alg(n_clusters=len(dataset) // args.cluster_sz).fit_predict(X)
    # balanced clustering is ultimate goal, below code does not seem to work for some reason
    # clusters, _ = kmeans_equal(X, num_clusters=len(dataset) // args.cluster_sz, cluster_size=args.cluster_sz)
    
    # compute cluster probabilities
    cscore = defaultdict(list)
    for idx, c in zip(ids, clusters):
        label = 1 if 'Yes' in dataset.get_target(idx) else 0
        cscore[c].append(label)
    csize = {c : len(arr) for c, arr in cscore.items()}
    cscore = {c : sum(arr) / len(arr) for c, arr in cscore.items()}
    print('Pseudo Labels: Cluster Probs =', describe(list(cscore.values())))
    print('Pseudo Labels: Cluster Sizes =', describe(list(csize.values())))

    # assign cluster probas to underlying dataset
    for idx, c in zip(ids, clusters):
        percent_proba = 100 * cscore[c]
        percent_proba = round(percent_proba)
        dataset.update_pseudo_label(idx, f'{percent_proba}%')
        if args.off_policy_policy_grad:
            reward = -compute_pseudo_loss([f'{percent_proba}%'], [dataset.get_target(idx)], reduction='none')
            dataset.update_importance_weight(idx, [-reward.item(), 1.0])

# same as in predict, consider updating that code block with this function call
def _load_pretrained_model(args):
    # load pretrained model
    if 'openai' in args.model:
        model = OpenAIModelWrapper(tokenizer, args)
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
                    **flash_args)
            print('loaded quantized model')
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(args.model)
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
                    **flash_args)
            print('loaded quantized model')
        else:
            if args.use_bfloat:
                model = transformers.AutoModelForCausalLM.from_pretrained(args.model, 
                    torch_dtype=torch.bfloat16)
            else:
                model = transformers.AutoModelForCausalLM.from_pretrained(args.model)

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

    return model

def _assign_pretrained_model_pseudo_labels(trloader, _, seed, args, accelerator=None):

    class ModelArgs:
        pass
    
    model_args = ModelArgs()

    # parse pretrain policy args and load policy
    if args.policy_checkpoint.endswith('/'):
        fname_to_parse = args.policy_checkpoint[:-1].split('/')[-1].split('.pt')[0]
    else:
        fname_to_parse = args.policy_checkpoint.split('/')[-1].split('.pt')[0]
    toks = fname_to_parse.split('~')
    model_args.trained = 1 # assume trained by default
    model_args.flash_attn = False
    model_args.use_bfloat = False
    model_args.checkpoint = args.policy_checkpoint
    model_args.device = accelerator.device
    for t in toks:
        k, v = t.split('=') if '=' in t else (None, None)
        if k == 'model':
            model_args.model = v.replace('+', '/')
        elif k == '4bit':
            model_args.load_in_4bit = int(v)
        elif k == 'lora':
            model_args.use_lora = int(v)
        elif k == 'trained':
            model_args.trained = int(v)
    policy = _load_pretrained_model(model_args)

    # manipulate underlying object
    dataset = trloader.dataset
    tokenizer = trloader.dataset.tokenizer
    dataset.sample_seed = seed # set seed so window samples match clustered
    trloader.dataset.with_label = False # ignore labels for policy evaluation
    trloader.dataset.change_prompt_style = True # change prompts for policy evaluation

    print('Pseudo Labels: Sampling policy')
    dataset.pseudo_labels = False # pseudo_labels may not be set yet, prevents error from being thrown
    trloader.dataset.importance_weights = False # importance weights may not have been set
    for batch in tqdm(trloader):

        # for now, we only support the SFT policy
        with torch.no_grad():
            output = policy(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            pos_token_idx = tokenizer(' Yes')['input_ids'][-1]
            tempered_logits = output['logits'] # no temp here, exps say temp is not needed on IID data
            probas = F.softmax(tempered_logits, dim=-1)[:, -1][:, pos_token_idx]

        for idx, proba in zip(batch['instance_idx'], probas):
            try:
                percent_proba = round(100 * proba.item())
            except ValueError:
                # model output Nan
                percent_proba = 50
            dataset.update_pseudo_label(idx, f'{percent_proba}%')
            reward = -compute_pseudo_loss([f'{percent_proba}%'], [dataset.get_target(idx)], reduction='none')
            dataset.update_importance_weight(idx, [-reward.item(), 1.0])
            # print(percent_proba, reward, dataset.get_target(idx))

    trloader.dataset.with_label = True # unignore labels
    trloader.dataset.change_prompt_style = False # change prompts for policy evaluation
    del policy # free up memory before any training starts
