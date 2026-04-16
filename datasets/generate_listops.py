""" GENERATE LISTOPS DATASET - CODE FROM https://github.com/google-research/long-range-arena/blob/main/lra_benchmarks/data/listops.py """

import os
import csv
import random as rd
import numpy as np
from pathlib import Path




TRAIN_SIZE = 96000
VAL_SIZE = 2000
TEST_SIZE = 2000
MIN_LEN = 500
MAX_LEN = 2000
MAX_DEPTH = 10
MAX_ARGS = 10

MIN = '[MIN'
MAX = '[MAX'
MED = '[MED'
FIRST = '[FIRST'
LAST = '[LAST'
SUM_MOD = '[SM'
END = ']'

OPERATORS = [MIN, MAX, MED, SUM_MOD]  # , FIRST, LAST]
VALUES = range(10)

VALUE_P = 0.25

DATA_DIR = 'data/listops_data'




def write_data(data, file_name):
    """Write data to a TSV file."""
    print(f"Writing {len(data)} samples to {file_name}")

    fp = Path(file_name).with_suffix(".tsv")
    fp.parent.mkdir(parents=True, exist_ok=True)

    with fp.open("w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["Source", "Target"])
        writer.writerows(data)

# Generates Synthetic Data
def generate_sample(depth, max_depth, max_args):

    if depth < max_depth:
        r = rd.random()
    else:
        r = 1

    if r > VALUE_P:
        value = rd.choice(VALUES)
        return value, 1
    else:
        length = 2
        num_values = rd.randint(2, max_args)
        values = []
        for _ in range(num_values):
            sub_t, sub_l = generate_sample(depth + 1, max_depth, max_args)
            values.append(sub_t)
            length += sub_l

            op = rd.choice(OPERATORS)
            t = (op, values[0])
            for value in values[1:]:
                t = (t, value)
            t = (t, END)
        return t, length
    
def generate_data(write=True):
    data = set()
    num_samples = TRAIN_SIZE + TEST_SIZE + VAL_SIZE
    while len(data) < num_samples:
        tree, length = generate_sample(1, MAX_DEPTH, MAX_ARGS)
        if length > MIN_LEN and length < MAX_LEN:
            data.add(tree)
            if len(data) % 1000 == 0:
                print('Processed {}'.format(len(data)))
    train = []
    for example in data:
        train.append([to_string(example), to_value(example)])

    val = train[TRAIN_SIZE:]
    test = val[VAL_SIZE:]
    val = val[:VAL_SIZE]
    train = train[:TRAIN_SIZE]

    print('Dataset size: %d/%d/%d' % (len(train), len(val), len(test)))

    if write:
        print("Writing data to file (this may take a while)...")
        write_data(train, DATA_DIR + '/train')
        write_data(val, DATA_DIR + '/val')
        write_data(test, DATA_DIR + '/test')
        print('Finished writing')

    return train, val, test


def rename_close_brackets(x: str) -> str:
    x = x.replace(']', 'X')
    x = x.replace('(', '')
    x = x.replace(')', '')
    return x

def whitespace_tokenize(text: str):
    return text.split()

def pad_sequence(seq, max_len, pad_val=0):
    if len(seq) > max_len:
        return seq[:max_len]
    l = seq + [pad_val] * (max_len - len(seq))
    return l

def to_string(t, parens=True):
    if isinstance(t, str):
        return t
    elif isinstance(t, int):
        return str(t)
    else:
        if parens:
            return '( ' + to_string(t[0]) + ' ' + to_string(t[1]) + ' )'

def to_value(t):
    """Compute the output of equation t.

    Args:
        t: a tree structure that represents equation t, list.

    Returns:
        The result of equation t, int.
    """
    if not isinstance(t, tuple):
        return t
    l = to_value(t[0])
    r = to_value(t[1])
    if l in OPERATORS:  # Create an unsaturated function.
        return (l, [r])
    elif r == END:  # l must be an unsaturated function.
        if l[0] == MIN:
            return min(l[1])
        elif l[0] == MAX:
            return max(l[1])
        elif l[0] == FIRST:
            return l[1][0]
        elif l[0] == LAST:
            return l[1][-1]
        elif l[0] == MED:
            return int(np.median(l[1]))
        elif l[0] == SUM_MOD:
            return np.sum(l[1]) % 10
    elif isinstance(l, tuple):
        # We've hit an unsaturated function and an argument.
        return (l[0], l[1] + [r])
    



if __name__ == '__main__':
    generate_data()