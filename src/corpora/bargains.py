import collections
import hashlib

from datasets import load_dataset
from src.corpora.utils import dump, basic_argparser, normalize_instances
from src.corpora.utils import build as build_utils

def parse_args():
    parser = basic_argparser()
    parser.add_argument('--title', action='store_true',
        help='Add item title to conversation history.')
    parser.add_argument('--seller_goal', action='store_true', default=False,
        help='Add seller goal to conversation history.')
    parser.add_argument('--buyer_goal', action='store_true', default=False,
        help='Add buyer goal to conversation history.')
    return parser.parse_args()

def main(args, persist=False):

    dataset = load_dataset("craigslist_bargains")
    formatted_instances = collections.defaultdict(list)

    for split in ['train', 'validation']:

        for row in dataset[split]:

            # print(row); exit()

            # if row['agent_info']['Role'][0] != 'buyer':
            #     print(row)
            #     # never prints

            # if row['items']['Title'][0] != row['items']['Title'][1]:
            #     print(row)
            #     # never prints

            try:
                final_price = [price for price in row['dialogue_acts']['price'] if price >= 0][-1]
            except IndexError:
                final_price = None
            targets = list(enumerate(row['agent_info']['Target']))

            acceptance_idx = None
            accept = None
            acts = False

            for i, intent in enumerate(row['dialogue_acts']['intent']):

                if intent:
                    acts = True

                if intent == 'accept' or intent == 'reject':
                    acceptance_idx = i

                    if intent != 'accept':
                        accept = 'A'
                    else:
                        values = [((-1)**i) * (goal - final_price) for i, goal in targets]
                        accept = 'B'
                        # output A if buyer gets a worse deal
                        secondary_output = 'Yes' if values[0] > values[1] else 'No'
                        
                    break

            if acts and accept == 'B':                
                utts = row['utterance'][:acceptance_idx]
                speakers = row['agent_turn'][:acceptance_idx]
                utts = [f'Speaker {s}: {u}' for u, s in zip(utts, speakers)]

                # meta info
                speaker_context = [
                    f'Situation: Speaker {i} has target price ${price}'
                    for i, price in targets
                    ]
                title = f'Situation: Item for sale is "{row["items"]["Title"][0]}"'


                if args.seller_goal:
                    utts = [speaker_context[1]] + utts

                if args.buyer_goal:
                    utts = [speaker_context[0]] + utts

                if args.title:
                    utts = [title] + utts

                inputs = ' || '.join(utts)
                if args.max_len is not None and len(inputs) > args.max_len:
                    inputs = '... ' + inputs[-args.max_len:]
                output = secondary_output
                formatted_instances[split].append({'input': inputs, 'target': output,
                    'situation' : [title] + speaker_context,
                    'instance_id': hashlib.md5(inputs.encode('utf-8')).hexdigest()})
    
    fname = 'bargains'

    if any([args.seller_goal, args.buyer_goal, args.title]):
        fname += '_'

    if args.seller_goal:
        fname += 's'
    if args.buyer_goal:
        fname += 'b'
    if args.title:
        fname += 't'

    formatted_instances = normalize_instances(formatted_instances, args.seed)
    if persist:
        dump(formatted_instances, args.dname, fname, args.seed)
    else:
        return formatted_instances

# to build the dataset on the fly, without persistence, use this
def build(split, seed=0, max_len=5000, write_situation=False):
    return build_utils(main, split, seed=seed, max_len=max_len,
        title=write_situation, seller_goal=write_situation, 
        buyer_goal=write_situation)

# to build the dataset and persist it, just call this file as main
if __name__ == '__main__':
    args = parse_args()
    main(args, persist=True)
