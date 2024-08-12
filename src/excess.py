import os
import json

if __name__ == '__main__':

    excess = []
    for file in os.listdir('outputs'):
        if 'excess' in file:
            x = json.load(open('outputs/' + file, 'r'))
            excess.extend(x.values())

    avg_summed_excess = sum(excess) / len(excess)
    print(f'Epsilon (sum) estimate:', avg_summed_excess)
    # we sample this value at T=1 before "simulated" temp scaling later on
    for T in [.25, .5, 1, 1.5, 1.75, 2, 2.5]:
        bound = lambda eps: abs(1 - (1 + eps) ** (1 / T))
        avg_bound = sum(map(bound, excess)) / len(excess)
        print(f'Bound (T={T}):', avg_bound)
    
