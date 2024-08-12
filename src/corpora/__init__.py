from src.corpora.awry import build as build_awry
from src.corpora.bargains import build as build_bargains
from src.corpora.casino import build as build_casino
from src.corpora.cmv import build as build_cmv
from src.corpora.deals import build as build_deals
from src.corpora.deleted import build as build_deleted
from src.corpora.donations import build as build_donations
from src.corpora.supreme import build as build_supreme

from src.corpora.utils import proba_prompt, hard_prompt

build_map = {
    'awry' : build_awry,
    'bargains' : build_bargains,
    'casino' : build_casino,
    'cmv' : build_cmv,
    'deals' : build_deals,
    'deleted' : build_deleted,
    'donations' : build_donations,
    'supreme' : build_supreme
}

# NOTE: inverted map, these are holdout sets
multi_build_map = {
    'final' : ['bargains'], # dropping bargains since it's not available atm
    'easy' : ['awry', 'deals'],
    'medium' : ['bargains', 'casino', 'deleted'],
    'hard' : ['cmv', 'donations', 'supreme'],
    # last one removes all big datasets for faster debugging
    'quick': ['bargains', 'awry', 'cmv', 'deals', 'deleted', 'supreme']
}

def build(name, split):

    # fixed args for all experiments
    seed = 1 # seed should always be 1
    max_len = None # downstream methods should resize on their own
    write_situation = False # deprecated, not used for any dataset

    # deprecated, shouldn't be used for any dataset
    # if name.endswith('+s'):
    #     write_situation = True
    #     name = name.split('+s')[0]

    if name in multi_build_map:
        allkeys = set(build_map.keys())
        holdout = set(multi_build_map[name])
        holdin = allkeys.difference(holdout)
        return {name : build(name, split) for name in holdin}
    else:
        return build_map[name](split, seed, max_len, write_situation)

prompt_average_map = {
    'awry' : 50,
    'bargains' : 69,
    'casino' : 72,
    'cmv' : 50,
    'deals' : 79,
    'deleted' : 56,
    'donations' : 54,
    'supreme' : 63
}

# treating data average as true mean
# for k,v in {k : sorted([round(x * 100) for i in [50] for x in binomtest(int(i * v / 100), i).proportion_ci(confidence_level=0.95)]) for k,v in prompt_average_map.items()}.items(): print(f"'{k}'",':',f'{v},')
noised_average_map = {
    'awry' : [36, 64],
    'bargains' : [53, 80],
    'casino' : [58, 84],
    'cmv' : [36, 64],
    'deals' : [64, 88],
    'deleted' : [41, 70],
    'donations' : [39, 68],
    'supreme' : [47, 75]
}

prompt_context_map = {
    'awry' : 'a group of Wikipedia contributors are deciding whether to retain the revisions made to an article',
    'bargains' : 'a buyer (Speaker 0) and seller (Speaker 1) are negotiating the price of an item',
    'casino' : 'the speakers are negotiating how to allocate the available resources among themselves',
    'cmv' : 'the speakers are defending their opinions on an issue',
    'deals' : 'the speakers are negotiating how to allocate the available resources among themselves',
    'deleted' : 'a group of Wikipedia contributors are deciding whether an article should be deleted',
    'donations' : 'one speaker is trying to persuade the other to donate to a charitable cause',
    'supreme' : 'a group of lawyers present Oral Arguments for a case that has been petitioned for review by the U.S. Supreme Court'
}

prompt_outcome_map = {
    'awry' : 'a personal attack',
    'bargains' : 'Speaker 0 (the buyer) getting a better deal',
    'casino' : 'both speakers satisfied with the deal',
    'cmv' : 'a personal attack',
    'deals' : 'the speakers making a deal',
    'deleted' : 'the article being deleted',
    'donations' : 'a donation',
    'supreme' : 'a favorable decision for the petitioner'
}

def prompt(name, proba=False, average=False, reminder=False):
    if name in multi_build_map:
        return True
    context = prompt_context_map[name]
    average = (average == 'low' and noised_average_map[name][0]) \
        or (average == 'high' and noised_average_map[name][1]) \
        or (average and prompt_average_map[name])
    outcome = prompt_outcome_map[name]
    fn = proba_prompt if proba else hard_prompt
    return fn(context, outcome, average=average, reminder=reminder)
