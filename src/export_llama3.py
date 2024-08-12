import os
import pandas as pd

from math import sqrt

if __name__ == '__main__':

    frames = os.listdir('frames')
    df = pd.concat([pd.read_csv(f'frames/{frame}') for frame in frames])
    df['SD'] = df['SVAR'].map(sqrt)
    df['n'] = 1 / 100

    df = df[df['EXP'].apply(lambda s: 'final' in s or 'gpt-4' in s)]
    parse = lambda k, s: s.split(f'{k}=')[-1].split('~')[0].split('&')[0]
    for k in ['model', 'instances', 'alg', '@avg']:
        df[k] = df['EXP'].apply(lambda s: parse(k, s))
    
    df = df[df['@avg'] == '1']
    print(df.groupby(['model', 'alg', 'instances']).agg('mean')['BS'])