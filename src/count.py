import json
import os

from collections import defaultdict

if __name__ == '__main__':
    counts = defaultdict(list)
    for file in os.listdir('outputs'):
        for delim in ['-test', '-val']:
            name = file.split(delim)[0]
            if name == file:
                continue
            name = name.split('=')[:-1]
            name = '='.join(name)
            n = len(json.load(open(f'outputs/{file}', 'r')))
            counts[name].append(n)
    for k,v in counts.items():
        # if len(v) < 56:
        print(k, len(v), sum(v) / len(v))
