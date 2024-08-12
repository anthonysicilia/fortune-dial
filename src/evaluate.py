import argparse
import json
import numpy as np
import pandas as pd
import random
import re
import os
import warnings

from sklearn.metrics import precision_recall_fscore_support as prf
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from scipy.optimize import root_scalar, minimize_scalar
from scipy.special import logit, expit

# living dangerously, comment this out to see the warnings
warnings.filterwarnings("ignore")
from src.utils import get_data, get_prompt, get_proba_from_string, maybe_str_or_int

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('instances', nargs="*", type=str)
    parser.add_argument('--find', type=str, default=None)
    parser.add_argument('--exclude', nargs="*")
    parser.add_argument('--val', type=str)
    parser.add_argument('--cal', nargs="+", type=int)
    parser.add_argument('--avg', nargs="+", type=maybe_str_or_int)
    parser.add_argument('--clf_stats', type=int, default=0)
    parser.add_argument('--match_avg_or_exclude_bss', type=int, default=0)
    parser.add_argument('--save_frames', type=int, default=1)
    return parser.parse_args()

swap_key = {'Yes' : 1, 'No' : 0}
swap = lambda x: swap_key[x]

def format(val, predictions, temp=None, verbose=True):

    y = []
    yhat = []
    uiids = []
    r = []
    keyerrors = 0

    for v in val:

        target = v['target']
        uiid = v['instance_id']

        try:
            pred = predictions[uiid]
        except KeyError:
            keyerrors += 1
            continue
        
        y.append(target)
        uiids.append(uiid)

        if type(pred) == float:
            if np.isnan(pred):
                pred = ''
            yhat.append(pred)
        elif type(pred) == str:
            yhat.append(get_proba_from_string(pred))
        else:
            raise TypeError('Unexpected format!')
    
    # track anything important about the formatting process
    a = set(v['instance_id'] for v in val)
    b = set(predictions.keys())
    if verbose:
        print('Missing (from val):', len(b.difference(a)))
        print('Total preds:', len(predictions))
        print('Missing (from preds):', '{}/{}'.format(keyerrors, len(val)))
    y = list(map(swap, y))
    if verbose:
        print('Corrupted probas:', sum([p == '' for p in yhat]))
    # validation loss uses 0.5 for unparsable responses; i.e., 
    # this may be a valid response from the model and should be
    # interpreted as such
    yhat = [p if p != '' else 0.5 for p in yhat]
    if verbose:
        print('Max/min proba:', max(yhat), min(yhat))
    yhat = [min(max(p, 0), 1) for p in yhat]

    if temp is not None:
        if type(temp) == float:
            yhat = [expit(logit(p) / temp) for p in yhat]
        else:
            # assume iterable
            yhat = [expit(temp[0] + logit(p) * temp[1]) for p in yhat]

    if verbose:
        return y, yhat, uiids
    else:
        return y, yhat

# Use expectation consistency to optimize temperature:
# https://github.com/SPOC-group/expectation-consistency/tree/master
def optimize_temp_ec(y, yhat):

    preds = [1 if p > 0.5 else 0 for p in yhat]
    error = sum(abs(a - b) for a, b in zip(y, preds)) / len(y)

    def objective(temp):
        probas = [expit(logit(p) / temp) for p in yhat]
        probas = [max(p, 1-p) for p in probas]
        return sum(probas) / len(probas) - (1.0 - error)
    try:
        res = root_scalar(objective, bracket=[0.01, 10])
        return float(res.root)
    except ValueError:
        return 1.

# standard temp scaling
def optimize_temp_ps(y, yhat, clipval=1000):
    yhat = np.array([logit(p) for p in yhat]).reshape(-1, 1)
    yhat = np.clip(yhat, -clipval, clipval) # expit 1000 is already 1.0
    clf = LogisticRegression(random_state=0).fit(yhat, y)
    return [clf.intercept_[0], clf.coef_[0][0]]

# standard temp scaling
def optimize_temp_ts(y, yhat):

    def objective(temp):
        probas = [expit(logit(p) / temp) for p in yhat]
        return brier_score_loss(y, probas)

    res = minimize_scalar(objective, bounds=[0.01, 10], method='bounded')
    return float(res.x)

if __name__ == '__main__':

    args = parse_args()

    print('Collecting data...')
    valset = get_data(args.val, 'val')
    testset = get_data(args.val, 'test')

    if not args.instances and args.find:
        not_excluded = lambda file: len(args.exclude) == 0 or all(e not in file for e in args.exclude)
        args.instances = [f'outputs/{file}' for file in os.listdir('outputs')
            if args.find in file and not_excluded(file)]

    for do_cal, avg in [(c, a) for c in args.cal for a in args.avg]:

        print('Cal:', do_cal)
        print('Avg:', avg)
        ref_proba = get_proba_from_string(get_prompt(args.val, proba=True, average=avg))
        print('Climatology Model:', ref_proba)
        
        for idx, instance in enumerate(args.instances):

            print('Starting', instance, '...')
            if do_cal:
                print('Optimizing temperature on val...')
                cal = instance.replace('-test', '-val')
                try:
                    if cal == instance:
                        raise FileNotFoundError("Shouldn't optim. with test set")
                    with open(cal, 'r') as file:
                        cal = json.load(file)
                    y, yhat, _ = format(valset, cal)
                    temp_selection = [
                        1., 
                        optimize_temp_ps(y, yhat), 
                        optimize_temp_ts(y, yhat), 
                        optimize_temp_ec(y, yhat)
                    ]
                    valscores = [
                        brier_score_loss(*format(valset, cal, temp=temp, verbose=False))
                        for temp in temp_selection
                    ]
                    temp = list(sorted(zip(temp_selection, valscores), key=lambda x: x[-1]))[0][0]
                except FileNotFoundError:
                    print('Skipping calibration for', instance)
                    temp = 1.
            else:
                temp = 1.

            try:
                with open(instance, 'r') as file:
                    predictions = json.load(file)
            except FileNotFoundError:
                print('Skipping prediction for', instance)
                continue

            y, yhat, uiids = format(testset, predictions, temp=temp)
            
            print('Instances', instance)
            print('Testing with temp:', temp)
            print('Testing with avg:', avg)
            # compute climatology model and BS stats
            cdata = [swap(x['target']) for x in valset]
            # c = sum(yi for yi in cdata) / len(cdata)
            # print('Climatology Model: P(y=1) = {}'.format(c))
            bs = brier_score_loss(y, yhat)
            cs = brier_score_loss(y, [ref_proba for _ in y])
            not_matched_or_unk = '@avg=' not in instance \
                or str(avg) != instance.split('@avg=')[1].split('~')[0]
            if args.match_avg_or_exclude_bss and not_matched_or_unk:
                bss = np.nan
            else:
                bss = 1 - (bs / cs)
            bias = sum(yhat) / len(yhat) - sum(y) / len(y)
            fk, ok = calibration_curve(y, yhat, n_bins=20, strategy='quantile')
            bin_size = len(y) / len(fk)
            cal = ((fk - ok) ** 2).sum() * (bin_size / len(y))
            # probabilisitc notion of variance, around ref proba
            var = ((ok - ref_proba) ** 2).sum() * (bin_size / len(y))
            # statistical notion of variance, around mean
            yhat_bar = sum(yhat) / len(yhat)
            svar = sum((yhati - yhat_bar) ** 2 for yhati in yhat) / len(yhat)
            print('BS, \tCS,\tBSS\tBIAS\tCAL\tVAR\tSVAR\tSZ')
            stats = ',\t'.join('{:.3f}'.format(x) for x in [bs, cs, bss, bias, cal, var, svar, bin_size])
            print(stats)
            if args.save_frames:
                exp = instance.strip('.json').split('/')[-1]
                exp += f'&cal={do_cal}&avg={avg}'
                pd.DataFrame({'EXP' : [exp], 'T' : [temp],
                    'BS' : [bs], 'CS' : [cs], 'BSS' : [bss], 
                    'BIAS' : [bias], 'CAL' : [cal], 'VAR' : [var], 
                    'SVAR' : [svar], 'SZ' : [bin_size]
                }).to_csv(f'frames/{exp}.json')
            # pd.DataFrame({'UIID' : uiids, 
            #     'Y' : y, 'YHAT' : yhat
            # }).to_csv(f'pp-outputs/{exp}.json')

            # compute accuracy and print prediction stats
            if args.clf_stats:
                yhat = [1 if p > 0.5 else 0 for p in yhat]
                acc =  1 - sum(abs(a - b) for a, b in zip(y, yhat)) / len(y)
                prec, recall, f1 = prf(y, yhat, average='binary', pos_label=1)[:3]
                print('ACC,\tPRE,\tREC,\tF1')
                stats = ',\t'.join('{:.1f}'.format(100 * x) for x in [acc, prec, recall, f1])
                print(stats)
                r = [random.choice([0, 1]) for _ in y]
                acc =  1 - sum(abs(a - b) for a, b in zip(y, r)) / len(y)
                prec, recall, f1 = prf(y, r, average='binary', pos_label=1)[:3]
                stats = ',\t'.join('{:.1f}'.format(100 * x) for x in [acc, prec, recall, f1])
                print(stats)
