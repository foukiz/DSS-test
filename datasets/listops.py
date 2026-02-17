import torch
import csv
import os
import pickle as pkl
import numpy as np
import random as rd
import pandas as pd

from pathlib import Path

try:
  from .dataset import Dataset
except ImportError:
  from dataset import Dataset

from torch.utils.data import TensorDataset




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




#class ListOps(Dataset):
class ListOps(Dataset):
    """Class to generate the ListOps dataset with some properties"""
    def __init__(
        self,
        train_size=96000,
        val_size=2000,
        test_size=2000,
        min_len=500,
        max_len=2000,
        max_depth=10,
        max_args=10,
        generate=False,
        preprocessed=True,
        data_dir='listops_data',
        **kwargs
    ):
        self.max_len = max_len
        self.min_len = min_len
        self.max_depth = max_depth
        self.max_args = max_args
        self.data_dir = data_dir
        self.generate = generate
        self.preprocessed = preprocessed
        self.vocab = None
       
        # reset after tests
        super().__init__(train_size, val_size, test_size, seq_length=max_len, **kwargs)

    @property
    def input_dimension(self):
        return (self._input_dimension,)
    
    @input_dimension.setter
    def input_dimension(self, value):
        self._input_dimension = value

    @property
    def input_flat_dimension(self):
        return self._input_dimension

    @property
    def image_size(self):
        return None

    @property
    def channels(self):
        return 1

    @property
    def num_outputs(self):
        return 10

    @property
    def test_size(self):
        return self.te_size

    @property
    def train_size(self):
        return self.tr_size

    @property
    def val_size(self):
        return self.va_size

    def get_train_ds(self):
        return self.train_ds

    def get_test_ds(self):
        return self.test_ds

    def get_val_ds(self):
        return self.val_ds
    
    def make_vocab(self):
        """ Vocab is made out of the tokens in the validation set """
        try:
          temp_val = pd.read_csv('listops_data/val.tsv', sep='\t', usecols=['Source'])
        except FileNotFoundError:
          raise FileNotFoundError('ListOps Data files were not generated yet')
        temp_val['Source'] = temp_val['Source'].apply(rename_close_brackets)
        temp_val['Source'] = temp_val['Source'].apply(whitespace_tokenize)

        vocab_set = set()
        for tokens in temp_val['Source']:
            vocab_set.update(tokens)
        vocab = {token: idx+1 for idx, token in enumerate(sorted(vocab_set))}
        vocab['<pad>'] = 0
        vocab['<unk>'] = len(vocab)
        print("vocab generated")
        self.input_dimension = len(vocab) - 1
        return vocab

    def write_data(self, data, file_name):
        """Write data to a TSV file."""
        print(f"Writing {len(data)} samples to {file_name}")

        fp = Path(file_name).with_suffix(".tsv")
        fp.parent.mkdir(parents=True, exist_ok=True)

        with fp.open("w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["Source", "Target"])
            writer.writerows(data)

    def preprocess_file2pkl(self):
        """ Extract data from tsv files, preprocess it and store it in pkl files """

        assert os.path.exists(self.data_dir + '/train.tsv') & os.path.exists(self.data_dir + '/val.tsv') & os.path.exists(self.data_dir + '/test.tsv'), (
            "Listops data was not generated yet. You might want to set generate to True?")

        # Construire vocab si nécessaire
        if self.vocab is None:
            self.vocab = self.make_vocab()

        df = pd.read_csv(self.data_dir + '/' + 'train.tsv', sep='\t', usecols=['Source','Target'])
        x_train, y_train = preprocess(df=df, vocab=self.vocab, max_len=self.max_len, kind='train')
        
        df = pd.read_csv(self.data_dir + '/' + 'val.tsv', sep='\t', usecols=['Source','Target'])
        x_val, y_val = preprocess(df=df, vocab=self.vocab, max_len=self.max_len, kind='val')
        
        df = pd.read_csv(self.data_dir + '/' + 'test.tsv', sep='\t', usecols=['Source','Target'])
        x_test, y_test = preprocess(df=df, vocab=self.vocab, max_len=self.max_len, kind='test')

        data_dic = {
          'train_ds': (x_train, y_train),
          'val_ds': (x_val, y_val),
          'test_ds': (x_test, y_test)
        }
        with open(self.data_dir + "/data.pkl", "wb") as f:
            pkl.dump(data_dic, f)

        self.preprocessed = True

    def make_data_out_of_pkl(self):
        assert os.path.exists(self.data_dir + '/data.pkl'), (
            "Listops data was not preprocessed yet. You might want to set preprocessed to False?")

        with open(self.data_dir + "/data.pkl", "rb") as f:
          data = pkl.load(f)

        return data

    def make_data_out_of_file(self, file_kind='train'):
        assert os.path.exists(self.data_dir + '/train.tsv') & os.path.exists(self.data_dir + '/val.tsv') & os.path.exists(self.data_dir + '/test.tsv'), (
              "Listops data was not generated yet. You might want to set generate to True?")

        # Construire vocab si nécessaire
        if self.vocab is None:
            self.vocab = self.make_vocab()

        df = pd.read_csv(self.data_dir + '/' + file_kind + '.tsv', sep='\t', usecols=['Source','Target'])
        inputs, targets = preprocess(df=df, vocab=self.vocab, max_len=self.max_len, kind=file_kind)

        return inputs, targets

    # Generates Synthetic Data
    def generate_sample(self, depth, max_depth, max_args):

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
                sub_t, sub_l = self.generate_sample(depth + 1, max_depth, max_args)
                values.append(sub_t)
                length += sub_l

                op = rd.choice(OPERATORS)
                t = (op, values[0])
                for value in values[1:]:
                    t = (t, value)
                t = (t, END)
            return t, length
        
    def generate_data(self, write=True):
        np.random.seed(42)
        data = set()
        num_samples = self.train_size + self.test_size + self.val_size
        while len(data) < num_samples:
            tree, length = self.generate_sample(1, self.max_depth, self.max_args)
            if length > self.min_len and length < self.max_len:
                data.add(tree)
                if len(data) % 1000 == 0:
                    print('Processed {}'.format(len(data)))
        train = []
        for example in data:
            train.append([to_string(example), to_value(example)])

        val = train[self.train_size:]
        test = val[self.val_size:]
        val = val[:self.val_size]
        train = train[:self.train_size]

        print('Dataset size: %d/%d/%d' % (len(train), len(val), len(test)))

        if write:
            self.write_data(train, self.data_dir + '/train')
            self.write_data(val, self.data_dir + '/val')
            self.write_data(test, self.data_dir + '/test')
            print('Finished writing all to file')
            self.generate = False
        
        return train, val, test

    def import_dataset(self):
        
        print("-" * 60 + f"Loading {type(self).__name__}" + "-" * 60)

        if self.generate:
            self.generate_data(write=True)

        if self.preprocessed:
            data = self.make_data_out_of_pkl()
            x_train, y_train = data['train_ds']
            x_val, y_val = data['val_ds']
            x_test, y_test = data['test_ds']

        else:
            x_train, y_train = self.make_data_out_of_file(file_kind='train')
            x_val, y_val = self.make_data_out_of_file(file_kind='val')
            x_test, y_test = self.make_data_out_of_file(file_kind='test')

        if self.vocab is None:
            self.vocab = self.make_vocab()

        self.input_dimension = len(self.vocab) - 1

        #train_ds = OneHotBatchDataset(self.input_flat_dimension, x_train, y_train)
        #val_ds = OneHotBatchDataset(self.input_flat_dimension, x_val, y_val)
        #test_ds = OneHotBatchDataset(self.input_flat_dimension, x_test, y_test)
        
        train_ds = TensorDataset(x_train, y_train)
        val_ds = TensorDataset(x_val, y_val)
        test_ds = TensorDataset(x_test, y_test)

        print("-" * 60 + f"{type(self).__name__} loaded" + "-" * 60)

        return train_ds, val_ds, test_ds





class OneHotBatchDataset(TensorDataset):

    def __init__(self, num_classes, *tensors: torch.Tensor) -> None:
        super().__init__(*tensors)
        self.num_classes = num_classes

    def __getitem__(self, index):
        batch_x, batch_y = super().__getitem__(index)
        # one hot encode the batched sequence to predict
        batch_x = torch.nn.functional.one_hot(batch_x, num_classes=self.num_classes)
        return (batch_x, batch_y)






""" PREPROCESSING AND UTILS FUNCTIONS """


NUM_WORKERS = os.cpu_count()  # nombre de threads pour DataLoader



def preprocess(df: pd.DataFrame, vocab: dict, max_len: int, kind='train'):

    assert kind in ('train', 'val', 'test'), "Arg file_kind should "
    "be one of 'train', 'val' or 'test', not {}".format(kind)

    assert list(df.columns) == ['Source', 'Target'], "Wrong DataFrame format ; should have columns "
    "'Source' and 'Target', but found {}".format(str(df.columns))

    print("preprocessing {} data...".format(kind))

    df['Source'] = df['Source'].apply(rename_close_brackets)
    df['Target'] = df['Target'].astype(int)

    # Tokenisation
    df['Source'] = df['Source'].apply(whitespace_tokenize)

    # Encoder les tokens en indices
    df['Source'] = df['Source'].apply(
        lambda tokens: [vocab.get(t, vocab['<unk>']) for t in tokens])
    
    df['Source'] = df['Source'].apply(
        lambda x: pad_sequence(x, max_len=max_len, pad_val=vocab['<pad>']))
    
    inputs = torch.tensor(df['Source'], dtype=torch.int64)
    targets = torch.tensor(df['Target'], dtype=torch.int64)

    print("preprocessing done.")

    return inputs, targets

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

    ds = ListOps(
        generate=True,
        preprocessed=False
    )
    ds.preprocess_file2pkl()
    print()