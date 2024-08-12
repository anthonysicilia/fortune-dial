import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from math import sqrt

from src.corpora import multi_build_map

if __name__ == '__main__':

    frames = os.listdir('frames')
    df = pd.concat([pd.read_csv(f'frames/{frame}') for frame in frames])
    df['SD'] = df['SVAR'].map(sqrt)
    df['n'] = 1 / 100

    df = df[df['EXP'].apply(lambda s: 'final' in s or ('alg=if-long' in s or 'alg=interp-long' in s))]
    parse = lambda k, s: s.split(f'{k}=')[-1].split('~')[0].split('&')[0]
    for k in ['model', 'instances', 'alg', '@avg']:
        df[k] = df['EXP'].apply(lambda s: parse(k, s))
    
    # drop non-sampling versions, which have low variance, and other failed run versions
    old_algs = df['alg'].apply(lambda s: s in ['explore', 'interp', 'exploit', 'exploit-smp', 'exploit2-smp'])
    df = df[~old_algs | ~(df['model'] == 'meta-llama+Llama-2-7b-chat-hf')]
    # use non exploded exploit
    # df['alg'] = df['alg'].apply(lambda s: s.replace('exploit2', 'exploit'))
    # for cases where we didn't explicitly note this; namely, llama alg=none
    df['if'] = df['EXP'].apply(lambda s: 'probs_for_instances' in s).astype(int).astype(str)
    # aggregate prior labels from @avg
    df['prior'] = df['@avg'].apply(lambda a: 'bad' if a in ['low', 'high'] else a)
    # capture both model and alg differences in model
    df['model'] = df['model'] + '~alg=' + df['alg']

    parse = lambda k, s: s.split(f'{k}=')[-1].split('&')[0]
    for k in ['cal', 'avg']:
        df[k] = df['EXP'].apply(lambda s: parse(k, s))

    # extract train sets for ft and pre-trained models
    df['ft_trainset'] = df['EXP'].apply(lambda s: s.split('~')[0].split('-')[0])
    df['pt_evalset'] = df['instances'].apply(lambda s: s.split('-')[0]).apply(
        lambda s: [k for k,v in multi_build_map.items() if k!='quick' and s in v].pop())
    is_pt = df['ft_trainset'].apply(lambda s: s.startswith('model'))
    trainset = []
    for pt, a, b in zip(is_pt, df['ft_trainset'], df['pt_evalset']):
        if pt:
            trainset.append(b)
        else:
            trainset.append(a)
    df['trainset'] = trainset

    trainslices = []
    testslices = []

    for k in multi_build_map.keys():
        if k == 'quick':
            continue
        trainset = df['trainset'] == k
        indomain = df['instances'].apply(lambda s: not any(v in s for v in multi_build_map[k]))
        trainslice = df[trainset & indomain]
        trainslices.append(trainslice)
        testslice = df[(is_pt | trainset) & ~indomain]
        testslices.append(testslice)
    
    # label and re group
    iid = pd.concat(trainslices)
    iid['iid'] = '1'
    ood = pd.concat(testslices)
    ood['iid'] = '0'
    df = pd.concat([iid, ood])

    # metrics we always look at and ways to aggregate them
    cols = ['BS', 'v70b', 'BSS', 'SD', 'n']
    agg = {c : 'mean' if c!= 'n' else 'sum' for c in cols}
    # useful masks
    processed = df['cal'] == '1'
    unprocessed = df['cal'] == '0'
    no_prior = df['prior'] == '0'
    nan_prior = df['prior'] == 'reminder'
    data_prior = df['prior'] == '1'
    bad_prior = df['prior'] == 'bad'
    iid = df['iid'] == '1'
    ood = df['iid'] == '0'

    # build the skill scores llama skill scores
    skill_models = [
        ('v7b', 'meta-llama+Llama-2-7b-chat-hf~alg=none'),
        ('v70b', 'meta-llama+Llama-2-70b-chat-hf~alg=none'),
        # no models are better than gpt4 except the explorer on hard (not signif.)
        # ('vGPT', 'openai+gpt-4~alg=none')
    ]
    for scol, name in skill_models:
        skillscores = []
        for i, row in df.iterrows():
            get_llama = (df['model'] == name) \
                & (df['instances'] == row['instances']) \
                & (df['cal'] == row['cal']) \
                & (df['if'] == '0')
            if len(df[get_llama]['BS'].unique()) > 1:
                print('Found a bad egg...')
                print(df[get_llama]); exit()
            llama_score = df[get_llama]['BS'].unique()[0]
            skillscores.append(1 - row['BS'] / llama_score)
        df[scol] = skillscores

    # 3 sig figs on 100pt scale used throughout our exps
    decimals = 1
    # Table 1
    models = [m for m in df['model'].unique() if 'alg=none' in m]
    models = df['model'].apply(lambda m: m in models)
    groups = (processed & no_prior) \
        | (processed & data_prior) \
        | (unprocessed & no_prior) \
        | (unprocessed & data_prior) \
        | (unprocessed & nan_prior) \
        | (unprocessed & bad_prior)
    print('Table 1: Baseline pre-trained models')
    print((df[models & groups].groupby(['iid', 'cal', 'model', 'if', 'prior'])[cols].agg(agg) * 100).round(decimals))

    # Table 2
    models = [m for m in df['model'].unique() if 'alg=none' not in m]
    models = models + [
        'meta-llama+Llama-2-7b-chat-hf~alg=none', 
        'TinyLlama+TinyLlama-1.1B-Chat-v0.6~alg=none',
        'HuggingFaceH4+zephyr-7b-beta~alg=none']
    models = df['model'].apply(lambda m: m in models)
    iid_group = (processed & data_prior & iid)
    ood_dev_group = (processed & data_prior & ood)
    zs_group = (unprocessed & bad_prior & ood)
    df['name'] = df['instances'].apply(lambda s: s.split('-')[0])
    long = df['name'].apply(lambda s: s in ['casino', 'supreme', 'deleted'])
    print('Table 2: Fine-tuned models')
    print((df[models & iid_group].groupby(['iid', 'cal', 'prior', 'model'])[cols].agg(agg) * 100).round(decimals))
    print((df[models & ood_dev_group].groupby(['iid', 'cal', 'prior', 'model'])[cols].agg(agg) * 100).round(decimals))
    print((df[models & zs_group].groupby(['iid', 'cal', 'prior', 'model'])[cols].agg(agg) * 100).round(decimals))
    print((df[models & zs_group].groupby(['iid', 'cal', 'prior', 'trainset', 'model'])[cols].agg(agg) * 100).round(decimals))

    # Table 2 Extended
    cbias = []
    for b,i in zip(df['BIAS'], df['instances']):
        if 'donations' in i or 'casino' in i:
            cbias.append(-b)
        elif 'awry' in i or 'cmv' in i:
            cbias.append(b)
        else:
            cbias.append(float('nan'))
    df['CBIAS'] = cbias
    print((df.groupby(['if', 'model'])[['CBIAS', 'BIAS', 'SD']].mean() * 100).round(decimals))
    # print('Ref')
    # print(ood[models & (ood['prior'] == 'bad') & (ood['cal'] == '0')].groupby(['baseset', 'model', 'cal', 'if', 'prior'])[['BS', 'BSS', 'VAR']].agg(agg) * 100)

    # # Table 2
    # print('Table 2: OOD results for fine-tuned models')
    # models = [m for m in df['model'].unique() if 'alg=none' not in m]
    # models = ood['model'].apply(lambda m: m in models)
    # # hasprior = ood['@avg'] != '0'
    # ood['zs'] = ((ood['cal'] == '0') & (ood['prior'] == 'bad')).astype(int).astype(str)
    # nonsense = (ood['zs'] == ood['cal']) # | ((ood['prior'] == 'bad') & (ood['cal'] == '1'))
    # print(ood[models & ~nonsense & (ood['zs'] == '0')].groupby(['zs', 'model', 'cal', 'if', 'prior'])[cols].agg(agg) * 100)
    # print(ood[models & ~nonsense & (ood['zs'] == '1')].groupby(['zs', 'trainset', 'model', 'cal', 'if', 'prior'])[cols].agg(agg) * 100)

    # # print('Ref 1: Base')
    # # models = [m for m in df['model'].unique() if 'alg=none' in m]
    # # models = ood['model'].apply(lambda m: m in models)
    # # models = models & (ood['if'] == '0')
    # # print(ood[models & ~nonsense & (ood['zs'] == '1')].groupby(['baseset', 'model', 'cal', 'if', 'prior'])[['BS', 'BSS', 'VAR']].agg(agg) * 100)

    # # Table 2
    # print('Ref 2: KL Comparison')
    # models = [m for m in df['model'].unique() if 'alg=none' not in m]
    # models = iid['model'].apply(lambda m: m in models)
    # insensible = (iid['cal'] == '0') & (iid['prior'] == 'bad')
    # print(iid[insensible].groupby(['trainset', 'model', 'cal', 'prior'])[cols].agg(agg) * 100)
    
    # # Table 2
    # print('Table 3: IID results for fine-tuned models')
    # models = [m for m in df['model'].unique() if 'alg=none' not in m]
    # models = iid['model'].apply(lambda m: m in models)
    # sensible = (iid['cal'] == '1') & (iid['prior'] == '1')
    # print(iid[sensible].groupby(['trainset', 'model', 'cal', 'prior'])[cols].agg(agg) * 100)

    # # model = ood['model'] == 'meta-llama+Llama-2-70b-chat-hf~alg=none'
    # # print(ood[model & (ood['if'] == '0') & (ood['baseset'] == 'hard') & (ood['cal'] == '0') & (ood['prior'] == 'bad')].sort_values(by=['instances', 'avg'])[['CS', 'BSS', 'instances', 'avg']])
    # # model = ood['model'] == 'meta-llama+Llama-2-7b-chat-hf~alg=explore'
    # # print(ood[model & (ood['if'] == '0') & (ood['baseset'] == 'hard') & (ood['cal'] == '0') & (ood['prior'] == 'bad')].sort_values(by=['instances', 'avg'])[['CS', 'BSS', 'instances', 'avg']])
    
    # # print('Overall Train')
    # # print(pd.concat(trainslices).groupby('model')[['BIAS', 'VAR']].agg(agg)) # 'VAR', 'CAL', 'BIAS'
    # # print()
    # # print('Overall Test')
    # # print(pd.concat(testslices).groupby('model')[['BS', 'BSS']].agg(agg)) # 'VAR', 'CAL', 'BIAS'
    # # print()
    # # print('Conflict')
    # # testslices = pd.concat(testslices)
    # # conflicts = [
    # #     ('awry', 1),
    # #     ('cmv', 1),
    # #     ('casino', -1),
    # #     ('deals', -1),
    # #     ('donations', -1)
    # # ]
    # # testslices['CBIAS'] = testslices.apply(lambda x: [c[1] * x['BIAS'] for c in conflicts if c[0] in x['instances']][0:1] or [None], axis=1)
    # # testslices['CBIAS'] = testslices['CBIAS'].map(lambda a: a.pop())
    # # print(testslices.groupby('model')[['CBIAS', 'BIAS', 'CAL', 'VAR']].agg(agg))
    




