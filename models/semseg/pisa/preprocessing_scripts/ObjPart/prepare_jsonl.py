import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

#TODO: This file is abandoned

def init(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path, help='input json file')
    parser.add_argument('--split-files', type=Path, nargs='+', help='paths to the .npy files containing the splits')
    parser.add_argument('--split-names', type=str, nargs='+', help='The output names for each split. Aligned to the names data in --split-files.')
    parser.add_argument('-o', '--output', type=Path, help='output jsonl.gz file')
    if args is not None:
        return parser.parse_args(args)
    return parser.parse_args()


def main(args):
    data = _load_data(args.input, '.json')
    tag2split = {}
    for fname, sname  in zip(args.split_files, args.split_names):
        img_list = _load_data(fname, '.npy')
        for img_tag in img_list:
            tag2split[img_tag] = sname
    
def separate_point_data(all_points, tag2split):
    def _flatten_label_data(label_d):
        ''' Separate so each entry has exactly one point
        '''
        for point_pos, labels in label_d.items():
            xcoor, ycoor =[int(coo) for coo in point_pos.split('_')]
            yield {'x': xcoor, 'y': }
    out_d = {}
    for point_tag, label_data in all_points.items():
        split = tag2split[point_tag]u
        if split not in out_d
            out_d[split] = []
        out_d[split].append()

def _validate_args(args):
    assert len(args.split_files) == len(args.split_names)

def _get_file_descriptor(input_f):
    if str(input_f) == '-' or str(input_f) == '/dev/stdin':
        return sys.stdin
    if input_f.suffix == '.gz':
        return gzip.open(str(input_f), 'rt', encoding='utf-8')
    if input_f.suffix == '.npy':
        return input_f.open('rb')
    return input_f.open('r', encoding='utf-8')

def _load_data(input_f, ext=None):
    fd = _get_file_descriptor(input_f)
    if ext is not None:
        if ext == '.json':
            return json.load(fd)
        elif ext == '.jsonl':
            return [json.loads(l) for l in fd]
        elif ext == '.npy':
            return np.load(fd)
        else:
            return [l for l in fd]

def _extract_data(input_f):
    '''Get the point labels from pascal_gt_clean.json'''
    if input_f == '-' or input_f == '/dev/stdin':
        return json.loads(sys.stdin)
    with open(input_f, 'r', encoding='utf-8') as f:
        return json.load(f)

if __name__ == '__main__':
    args = init()
    main(args)
