from src.corpora.utils import basic_argparser
from src.corpora.utils import build as build_utils
from src.corpora.convokit import ConvoKitFormatter

class ChangeFormatter(ConvoKitFormatter):

    def drop_label(self, utts):
        return utts[:-1]
    
    def format_utt(self, utt):
        return utt.strip()
    
    def format_output(self, convo, order):
        return 'Yes' if convo.retrieve_meta('has_removed_comment') \
            else 'No'

def parse_args():
    parser = basic_argparser()
    return parser.parse_args()

def main(args, persist=False):
    formatter = ChangeFormatter(
        "conversations-gone-awry-cmv-corpus",
        seed=args.seed,
        max_len=args.max_len)
    if persist:
        formatter.dump(args.dname, 'cmv')
    else:
        return formatter.formatted_instances

# to build the dataset on the fly, without persistence, use this
def build(split, seed=0, max_len=5000, write_situation=False):
    return build_utils(main, split, seed=seed, max_len=max_len)

# to build the dataset and persist it, just call this file as main
if __name__ == '__main__':
    args = parse_args()
    main(args, persist=True)