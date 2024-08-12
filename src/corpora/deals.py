import argparse
import collections
import hashlib
import json
import numpy as np
import os

from pathlib import Path

from datasets import load_dataset
from src.corpora.utils import dump, basic_argparser, normalize_instances
from src.corpora.utils import build as build_utils

def parse_args():
    parser = basic_argparser()
    parser.add_argument('--write_situation', action='store_true',
        help='Write the situation information to the conversation.')
    return parser.parse_args()

def get_item(num, index):
    item = ['book', 'hat', 'ball'][index]
    if num != 1:
        return str(num) + ' ' + item + 's'
    else:
        return str(num) + ' ' + item

def get_items(counts):
    return f'{get_item(counts[0], 0)}, {get_item(counts[1], 1)}, and {get_item(counts[2], 2)}'

def get_values(value):
    return f'books at {value[0]} points, hats at {value[1]} points, and balls at {value[2]} points.'

def compute_value(speaker):
    return sum(int(o) * int(v) for o,v in zip(speaker['outcome'], speaker['value']))

def main(args, persist=False):

    dataset = load_dataset("deal_or_no_dialog")
    formatted_instances = collections.defaultdict(list)
    kwords = ['disagree', 'no_agreement']

    for split in ['train', 'test', 'validation']:
        for row in dataset[split]:
            inputs = row['dialogue'] \
                .replace('THEM:', '|| Speaker 0:') \
                .replace('YOU:', '|| Speaker 1:') \
                .replace('<eos>', '') \
                .lstrip('|| ') \
                .rstrip('Speaker 0: <selection>') \
                .rstrip('Speaker 1: <selection>') \
                .rstrip(' ||')
            # output A if speakers come to a deal
            output = 'No' if any(k in row['output'] for k in kwords) else 'Yes'
            speaker_inputs = [row['partner_input'], row['input']]
            speaker_context = [f'Situation: {get_items(x["count"])} are available. '
                f'Speaker {i} values {get_values(x["value"])}'
                for i, x in enumerate(speaker_inputs)]
            if args.write_situation:
                s = ' || '.join(speaker_context)
                inputs = f'{s} || {inputs}'
            # use only whether deal occurs for this dataset
            # if output == 'B': # need to swtich this to 'A' if brought back
            #     outcomes = [int(x.split(' ')[0]) for x in row['output'].split('=')[1:]]
            #     speaker_inputs[1]['outcome'] = outcomes[:3]
            #     speaker_inputs[0]['outcome'] = outcomes[3:]
            #     speaker_value = [compute_value(speaker) for speaker in speaker_inputs]
            #     second_output = '1' if speaker_value[0] < speaker_value[1] else '0'
            #     output = f'B {second_output}'
            formatted_instances[split].append({'input': inputs, 'target': output,
                'situation' : speaker_context, 
                'instance_id': hashlib.md5(inputs.encode('utf-8')).hexdigest()})
    
    fname = 'deals'
    if args.write_situation:
        fname += '_s'
    formatted_instances = normalize_instances(formatted_instances, args.seed)
    if persist:
        dump(formatted_instances, args.dname, fname, args.seed)
    else:
        return formatted_instances

# to build the dataset on the fly, without persistence, use this
def build(split, seed=0, max_len=5000, write_situation=False):
    return build_utils(main, split, seed=seed, max_len=max_len,
        write_situation=write_situation)

# to build the dataset and persist it, just call this file as main
if __name__ == '__main__':
    args = parse_args()
    main(args, persist=True)