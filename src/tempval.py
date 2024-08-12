import json
import os

from math import log, exp
from scipy.optimize import minimize_scalar, minimize
from tqdm import tqdm

from src.utils import get_data
from src.evaluate import format, optimize_temp_ec, optimize_temp_ps, optimize_temp_ts
from src.evaluate import brier_score_loss


def tien_scaling(y, yhat):

    # recreation of: https://aclanthology.org/2023.emnlp-main.330.pdf
    # The primary specifications in the paper include:
    # - fitting a single parameter b
    # - so the result is proportional to p^b
    # Note, exp(b log(p)) = exp(log(p^b))) = p^b
    # p^b is a valid probability too (in unit interval), so we just use this;
    # there are other possible choices for constants of proportionality k
    # that keep k * p^b a valid proba. but it's not clear which the orig. paper means (if any).
    # So, we also provide a modified 2 parameter version (tien_scaling2param), 
    # which fits k optimally

    def objective(temp):
        probas = [exp(log(p) / temp) for p in yhat]
        return brier_score_loss(y, probas)

    res = minimize_scalar(objective, bounds=[0.01, 10], method='bounded')
    return float(res.fun)

def tien_scaling2param(y, yhat):

    # we explore a two parameter version of the proposal by tien.
    # so, this is still proportional to p^b but fits the constant of proportionality,
    # acting similiarly to the bias correction term in our method. 
    # our motivating analysis is still not 
    # necessarily valid for this scaling procedure

    def objective(args):
        temp, k = args
        probas = [exp(log(p) / temp) / k for p in yhat]
        return brier_score_loss(y, probas)

    # k >= 1 keeps k * p^b proportional and still valid proba.
    res = minimize(objective, [1., 1.], bounds=[[0.01, 10], [1, 10]])
    return float(res.fun)

if __name__ == '__main__':

    cache = dict()
    best = dict()
    one = dict()
    tien = dict()
    tien2 = dict()

    for file in tqdm(os.listdir('outputs')):

        if 'tempval@' in file:
            trainset = file.split('~')[0]
            temp = float(file.split('temp=')[-1].strip('.json'))
            instances = file.split('instances=')[-1].split('-tempval')[0]
            if instances not in cache:
                cache[instances] = get_data(f'build:{instances}', 'val')
            with open(f'outputs/{file}', 'r') as file:
                cal = json.load(file)
            y, yhat = format(cache[instances], cal, verbose=False)
            bs = brier_score_loss(y, yhat)
            model = instances + '-' + trainset
            if model not in best or best[model][-1] > bs:
                best[model] = (temp, bs)
                
            if temp == 1 and model not in one:
                temp_selection = [
                    1., 
                    optimize_temp_ps(y, yhat), 
                    optimize_temp_ts(y, yhat), 
                    optimize_temp_ec(y, yhat)
                ]
                valscores = [
                    brier_score_loss(*format(cache[instances], cal, temp=temp, verbose=False))
                    for temp in temp_selection
                ]
                minscore = list(sorted(zip(temp_selection, valscores), key=lambda x: x[-1]))[0][1]
                one[model] = minscore
                tien[model] = tien_scaling(y, yhat)
                tien2[model] = tien_scaling2param(y, yhat)

    avg_bs_diff = []
    avg_gain_t = []
    avg_gain_t2 = []
    avg_gain_k = []
    avg_gain_o = []
    avg_gain_o2 = []
    alt_choice_k = 0
    alt_choice_t = 0
    alt_choice_t2 = 0
    n = 0
    print('Setup, pref K, pref T')
    for model, bs1 in one.items():

        # instances, trainset = model.split('-')
        # if instances in multi_build_map[trainset]:
        #     # we can't scale for ood evaluation anyway
        #     continue

        temp2, bs2 = best[model]
        bs3 = tien[model]
        bs4 = tien2[model]

        if bs2 < bs1:
            alt_choice_k += 1
            avg_gain_k.append(bs1 - bs2)

        if bs3 < bs1:
            alt_choice_t += 1
            avg_gain_t.append(bs1 - bs3)

        if bs4 < bs1:
            alt_choice_t2 += 1
            avg_gain_t2.append(bs1 - bs4)

        if bs1 < min(bs2, bs3):
            avg_gain_o.append(min(bs2, bs3) - bs1)
        
        if bs1 < bs4:
            avg_gain_o2.append(bs4 - bs1)

        n += 1
    
    print('Perc. alt choice K:', alt_choice_k / n)
    if avg_gain_k:
        print('\tgain:', sum(avg_gain_k) / len(avg_gain_k))
    print('Perc. alt choice T:', alt_choice_t / n)
    if avg_gain_t:
        print('\tgain:', sum(avg_gain_t) / len(avg_gain_t))
    print('Perc. alt choice T2:', alt_choice_t2 / n)
    if avg_gain_t2:
        print('\tgain:', sum(avg_gain_t2) / len(avg_gain_t2))
        
    if avg_gain_o:
        print('\tgain:', sum(avg_gain_o) / len(avg_gain_o))
    if avg_gain_o2:
        print('\tgain:', sum(avg_gain_o2) / len(avg_gain_o2))