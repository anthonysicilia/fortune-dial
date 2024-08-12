import argparse
import json
import os
import random
import re

from pathlib import Path

PROBA_Q = 'What is the percent probability that this specific conversation will end with'

def proba_prompt(context, outcome, average=None, reminder=False):
    prompt = '[Segment Start]\n'
    prompt += '[arg]\n'
    prompt += '[Segment End]\n\n'
    prompt += f'In the preceding conversation segment, {context}. '
    if average:
        prompt += f'On average, this type of conversation ends with {outcome} about {average}% of the time. '
    elif reminder:
        prompt += f'Remember, these are real humans talking and they might not perfectly cooperate or agree about conflict. '
        prompt += f'The conversation may end with {outcome}, or not. '
    prompt += f'{PROBA_Q} {outcome}?'
    return prompt

def hard_prompt(context, outcome, average=None, reminder=False):
    prompt = '[Segment Start]\n'
    prompt += '[arg]\n'
    prompt += '[Segment End]\n\n'
    prompt += f'In the preceding conversation segment, {context}. '
    if average:
        prompt += f'On average, this type of conversation ends with {outcome} about {average}% of the time. '
    elif reminder:
        prompt += f'Remember, these are real humans talking and they might not perfectly cooperate or agree about conflict. '
        prompt += f'The conversation may end with {outcome}, or not. '
    prompt += f'Will this specific conversation end with {outcome}?'
    return prompt

def natsort(arr):
    """
    lt weight implementation of natsort
    https://github.com/SethMMorton/natsort/wiki/How-Does-Natsort-Work%3F
    """

    def int_map(x):
        try:
            return int(x)
        except:
            return x
    
    sortable = []

    for elem in arr:
        sortable.append(list(map(int_map, re.split(r'(\d+)', elem))))
    
    arr = list(zip(arr, sortable))
    
    return [x[0] for x in sorted(arr, key=lambda y: y[1])]

def shuffle_dict_elems(data_dict, seed):
    # shuffle splits independently (maintains splits)
    random.seed(seed)
    for k in sorted(data_dict.keys()):
        random.shuffle(data_dict[k])

def normalize_instances(instances, seed):
    random.seed(seed)
    # check for easy fixes before wiping and making a new split
    if 'validation' in instances.keys():
        instances['val'] = instances['validation']
        del instances['validation']
    # wipe and make new split if not normal
    normalkeys = set(['train', 'test', 'val'])
    normal = normalkeys == set(instances.keys())
    if not normal:
        allinstances = []
        for v in instances.values():
            allinstances.extend(v)
        random.shuffle(allinstances)
        n = len(allinstances)
        vstart = int(n * 0.7)
        vend = int(n * 0.7) + int(n * 0.15)
        instances = {
            'train' : allinstances[:vstart],
            'val' : allinstances[vstart:vend],
            'test' : allinstances[vend:]
        }
    
    # make sure data is shuffled here, so any downstream subsets are guaranteed random
    shuffle_dict_elems(instances, seed)
    return instances
    
def dump(instances, dname, fname, seed):
    random.seed(seed)
    for k, v in instances.items():
        random.shuffle(v)
        print(k, len(v))
        Path(dname).mkdir(parents=True, exist_ok=True)
        path = os.path.join(dname, f'{fname}_{k}.jsonl')
        with open(path, 'w') as f:
            f.write('\n'.join([json.dumps(d) for d in v]))

def basic_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dname', type=str, default='data',
        help='Base directory name to save under.')
    parser.add_argument('--seed', type=int, default=0,
        help='Random seed for data splits.')
    parser.add_argument('--max_len', type=int, default=None,
        help='Max char. length of dialogue, excluding chars from meta. info.')
    return parser

def build(main, split, seed=0, max_len=5000, **kwargs):
    args = argparse.Namespace(seed=seed, max_len=max_len, **kwargs)
    return main(args)[split]
