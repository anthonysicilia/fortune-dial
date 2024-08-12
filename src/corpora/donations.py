from src.corpora.convokit import ConvoKitFormatter
from src.corpora.utils import basic_argparser
from src.corpora.utils import build as build_utils

class DonationsFormatter(ConvoKitFormatter):

    def drop_label(self, utts):
        return utts
    
    def format_utt(self, utt):
        return utt.strip()
    
    def format_output(self, convo, order):
        amount = convo.retrieve_meta('donation_ee')
        return f'Yes' if amount > 0.0 else 'No'
    
    def get_internal_speaker_id(self, utt):
        return utt.meta['role']
    
    def situation(self, convo, order):
        i = order.index(0)
        j = order.index(1)
        c = f'Situation: Speaker {i} is persuading Speaker {j}.'
        context = [None, None]
        context[i] = c
        return context

def parse_args():
    parser = basic_argparser()
    parser.add_argument('--write_situation', action='store_true',
        help='Add situation to conversation history.')
    return parser.parse_args()

def main(args, persist=False):
    formatter = DonationsFormatter(
        "persuasionforgood-corpus",
        seed=args.seed,
        max_len=args.max_len,
        write_situation=args.write_situation)
    if persist:
        fname = 'donations'
        if args.write_situation:
            fname += '_s'
        formatter.dump(args.dname, fname)
    else:
        return formatter.formatted_instances

# to build the dataset on the fly, without persistence, use this
def build(split, seed=0, max_len=5000, write_situation=False):
    return build_utils(main, split, seed=seed, max_len=max_len,
        write_situation=write_situation)

# to build the dataset and persist it, just call this file as main
if __name__ == '__main__':
    args = parse_args()
    main(args, persist=True)