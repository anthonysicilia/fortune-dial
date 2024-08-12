from src.corpora.convokit import ConvoKitFormatter
from src.corpora.utils import basic_argparser
from src.corpora.utils import build as build_utils

class CasinoFormatter(ConvoKitFormatter):

    def drop_label(self, utts):
        kwords = ['Submit-Deal', 'Walk-Away']
        for i, utt in enumerate(utts):
            if any(k in utt.text for k in kwords):
                return utts[:i]
    
    def format_utt(self, utt):
        return utt.strip()
    
    def format_output(self, convo, order):
        unhappy = False
        for v in convo.retrieve_meta('participant_info').values():
            # {'Extremely dissatisfied', 'Slightly dissatisfied', 'Slightly satisfied', 'Undecided', 'Extremely satisfied'}
            outcome = v['outcomes']['satisfaction']
            unhappy = unhappy or 'dissatisfied' in outcome or 'Undecided' in outcome
        return 'Yes' if not unhappy else 'No'
    
    def get_internal_speaker_id(self, utt):
        return utt.meta['speaker_internal_id']
    
    def situation(self, convo, order):
        contexts = []
        for i, speaker in enumerate(order):
            priors = {v : k 
                for k,v in convo.retrieve_meta('participant_info') \
                    [speaker]['value2issue'].items()}
            c = f'Situation: Speaker {i} values food with {priors["Food"]} priority, ' \
                f'water with {priors["Water"]} priority, ' \
                f'and firewood with {priors["Firewood"]} priority.'
            contexts.append(c)
        return contexts
    
    def speaker_value(self, convo, order):
        values = []
        for speaker in order:
            values.append(convo.retrieve_meta('participant_info')[speaker]['outcomes']['points_scored'])
        return map(str, values)

def parse_args():
    parser = basic_argparser()
    parser.add_argument('--write_situation', action='store_true',
        help='Add situation to conversation history.')
    return parser.parse_args()

def main(args, persist=False):
    formatter = CasinoFormatter(
        "casino-corpus",
        seed=args.seed,
        max_len=args.max_len,
        write_situation=args.write_situation)
    if persist:
        fname = 'casino'
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