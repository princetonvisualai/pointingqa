'''Used to sanity check the HIT tasks distributed in this folder'''
import os
import json
import collections


def obj2key(obj):
    key = obj.keys()[0]
    data = obj[key]['points']['data'].keys()[0]
    dat = data.split('_')
    return key + "_" + dat[0] + "_" + dat[1] + "_" + dat[2] + "_" + \
        str(obj[key]['points']['data'][data]['points'][0][0]) + "_" + \
        str(obj[key]['points']['data'][data]['points'][0][1])


def main():
    frequencies = collections.defaultdict(int)
    for fname in os.listdir(os.path.dirname(os.path.abspath(__file__))):
        if 'BYNAME' not in fname:
            continue
        with open(fname) as f:
            for i, l in enumerate(f.readlines()):
                HIT = json.loads(l.strip())
                for obj in HIT:
                    frequencies[obj2key(obj)] += 1
        relative_frequencies = collections.defaultdict(int)
        for key, freq in frequencies.items():
            relative_frequencies[freq] += 1
    print(relative_frequencies)
#    print(compare_data)


if __name__ == "__main__":
    main()
